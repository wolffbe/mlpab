"""Full FTI pipeline for PM2.5 forecasting — runs ON the Hopsworks platform as a job.

Feature -> Training -> Inference:
  1. Read raw history + forecast CSVs (staged on HopsFS).
  2. Engineer features (weather + pm25 lag + calendar) into feature group `airq31cf6a`.
  3. Build feature view + training dataset `airqtd31cf6a`.
  4. Train a PM2.5 regressor, evaluate (RMSE), register as `airqmodel31cf6a` with metrics.
  5. Predict every forecast row into online+offline feature group `airqpred31cf6a`.
"""
import os
import numpy as np
import pandas as pd
import joblib

import hopsworks
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

FEATURES = [
    "pm25_lag1", "temperature", "humidity", "wind_speed",
    "pressure", "precipitation", "precip_log", "month", "doy_sin", "doy_cos",
]


def engineer(df):
    """Model-independent feature engineering computable from a single row.

    Only uses signals present in BOTH history and forecast (weather + pm25_lag1 +
    calendar derived from `date`) so the model trains and serves on the same shape.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month.astype("int64")
    doy = df["date"].dt.dayofyear.astype("float64")
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for c in ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]:
        df[c] = df[c].astype("float64")
    df["precip_log"] = np.log1p(df["precipitation"].clip(lower=0))
    # keep `date` as datetime64 — required for use as a DATE/TIMESTAMP event_time
    return df


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dsapi = project.get_dataset_api()
    mr = project.get_model_registry()

    # --- 1. Pull staged raw CSVs from HopsFS into the job's local workspace ---
    for fn in ["airquality_history.csv", "forecast_days.csv"]:
        if os.path.exists(fn):
            os.remove(fn)
        dsapi.download(f"Resources/airq/{fn}", fn, overwrite=True)

    hist = pd.read_csv("airquality_history.csv")
    fcst = pd.read_csv("forecast_days.csv")
    print(f"[data] history={hist.shape} forecast={fcst.shape}")

    hist_e = engineer(hist)          # includes pm25 label
    fcst_e = engineer(fcst)          # no pm25

    # --- 2. Feature group with engineered features ---
    fg = fs.get_or_create_feature_group(
        name="airq31cf6a",
        version=1,
        description="Engineered daily air-quality features (weather + pm25 lag + calendar) with pm25 target.",
        primary_key=["date"],
        event_time="date",
        online_enabled=False,
    )
    fg_cols = ["date"] + FEATURES + ["pm25"]
    fg.insert(hist_e[fg_cols], write_options={"wait_for_job": True})
    print("[fg] airq31cf6a inserted")

    # --- 3. Feature view + training dataset (artifact named airqtd31cf6a) ---
    try:
        try:
            existing = fs.get_feature_view(name="airqtd31cf6a", version=1)
            existing.delete()
        except Exception:
            pass
        query = fg.select([*FEATURES, "pm25"])
        fv = fs.get_or_create_feature_view(
            name="airqtd31cf6a",
            version=1,
            description="Feature view + training dataset for the airq PM2.5 regressor.",
            query=query,
            labels=["pm25"],
        )
        try:
            fv.create_training_data(write_options={"wait_for_job": True})
            print("[td] airqtd31cf6a materialized")
        except Exception as e:
            print(f"[td] materialize warning: {e}")
    except Exception as e:
        print(f"[fv] warning: {e}")
        fv = None

    # --- 4. Train + evaluate + register ---
    data = hist_e.sort_values("date").reset_index(drop=True)
    X = data[FEATURES].values
    y = data["pm25"].values.astype("float64")

    # time-ordered holdout for an honest metric
    n = len(data)
    cut = int(n * 0.8)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    # Candidate estimators. Linear models are scaled (Ridge is scale-sensitive);
    # tree ensembles are scale-invariant. Selection is by 5-fold CV RMSE.
    def scaled(est):
        return Pipeline([("scaler", StandardScaler()), ("model", est)])

    candidates = {}
    for alpha in [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        candidates[f"ridge_a{alpha}"] = scaled(Ridge(alpha=alpha))
    candidates["rf"] = RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=2,
        max_features=0.6, random_state=42, n_jobs=-1)
    candidates["gbr"] = GradientBoostingRegressor(
        n_estimators=500, max_depth=2, learning_rate=0.03,
        subsample=0.8, random_state=42)
    candidates["hgb"] = HistGradientBoostingRegressor(
        max_iter=600, max_depth=3, learning_rate=0.05,
        l2_regularization=1.0, random_state=42)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    from sklearn.base import clone
    for name, est in candidates.items():
        cv_rmse = -cross_val_score(
            est, X, y, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1
        ).mean()
        scores[name] = float(cv_rmse)
        print(f"[eval] {name} CV RMSE={cv_rmse:.4f}")

    best_name = min(scores, key=scores.get)
    best_cv_rmse = scores[best_name]

    # Honest time-ordered holdout for the registered metric.
    holdout = clone(candidates[best_name])
    holdout.fit(Xtr, ytr)
    hold_rmse = float(np.sqrt(mean_squared_error(yte, holdout.predict(Xte))))
    base_rmse = float(np.sqrt(mean_squared_error(yte, np.full_like(yte, ytr.mean()))))
    print(f"[eval] best={best_name} CV={best_cv_rmse:.4f} time-holdout={hold_rmse:.4f} "
          f"(baseline={base_rmse:.4f})")

    # refit the winning estimator on ALL history for the deployed/forecast model
    final = clone(candidates[best_name])
    final.fit(X, y)

    model_dir = "airq_model_dir"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final, os.path.join(model_dir, "model.pkl"))

    input_example = data[FEATURES].head(1).to_dict(orient="records")[0]
    metrics = {
        "rmse": hold_rmse,
        "cv_rmse": best_cv_rmse,
        "baseline_rmse": base_rmse,
    }
    desc = f"PM2.5 daily regressor ({best_name}). CV RMSE={best_cv_rmse:.4f}, time-holdout RMSE={hold_rmse:.4f}."
    try:
        model = mr.sklearn.create_model(
            name="airqmodel31cf6a",
            metrics=metrics,
            description=desc,
            input_example=input_example,
            feature_view=fv,
        )
        model.save(model_dir)
    except Exception as e:
        print(f"[model] sklearn-flavor register failed ({e}); retrying python flavor")
        model = mr.python.create_model(
            name="airqmodel31cf6a",
            metrics=metrics,
            description=desc,
            input_example=input_example,
        )
        model.save(model_dir)
    print(f"[model] registered airqmodel31cf6a metrics={metrics}")

    # --- 5. Predict forecast rows -> online+offline prediction FG ---
    fpred = final.predict(fcst_e[FEATURES].values).astype("float64")
    out = pd.DataFrame({
        "date": pd.to_datetime(fcst_e["date"]).dt.strftime("%Y-%m-%d").values,
        "pm25_pred": fpred,
    })
    print(f"[predict] {out.shape} rows; head:\n{out.head().to_string(index=False)}")

    pred_fg = fs.get_or_create_feature_group(
        name="airqpred31cf6a",
        version=1,
        description="PM2.5 predictions for forecast days (online + offline).",
        primary_key=["date"],
        online_enabled=True,
    )
    pred_fg.insert(out, write_options={"wait_for_job": True})
    print("[fg] airqpred31cf6a inserted (online+offline)")

    print("DONE_OK")


if __name__ == "__main__":
    main()
