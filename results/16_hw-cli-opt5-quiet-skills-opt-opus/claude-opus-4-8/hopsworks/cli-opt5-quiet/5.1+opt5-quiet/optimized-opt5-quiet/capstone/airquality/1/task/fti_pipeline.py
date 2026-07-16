"""Full FTI pipeline for PM2.5 forecasting — runs as a Hopsworks Job (platform-side)."""
import os
import numpy as np
import pandas as pd
import joblib
import hopsworks

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES = [
    "pm25_lag1", "temperature", "humidity", "wind_speed",
    "pressure", "precipitation", "month", "dayofyear",
]

def engineer(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month.astype("int64")
    df["dayofyear"] = df["date"].dt.dayofyear.astype("int64")
    for c in ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]:
        df[c] = df[c].astype("float64")
    return df

def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    dapi = project.get_dataset_api()

    # ---- download inputs uploaded to HopsFS ----
    for f in ["airquality_history.csv", "forecast_days.csv"]:
        if os.path.exists(f):
            os.remove(f)
        dapi.download(f"Resources/airq/{f}", local_path=".", overwrite=True)

    hist = engineer(pd.read_csv("airquality_history.csv"))
    fcst = engineer(pd.read_csv("forecast_days.csv"))

    # ================= FEATURE GROUP =================
    feat_cols = ["date"] + FEATURES + ["pm25"]
    fg = fs.get_or_create_feature_group(
        name="airqf3a8d0", version=1,
        description="Engineered air-quality + weather features",
        primary_key=["date"], event_time="date", online_enabled=False,
    )
    fg.insert(hist[feat_cols], wait=True)
    print("FEATURE GROUP inserted:", len(hist))

    # ================= FEATURE VIEW + TRAINING DATASET =================
    query = fg.select(FEATURES + ["pm25"])
    fv = fs.get_or_create_feature_view(
        name="airqtdf3a8d0", version=1, query=query, labels=["pm25"],
    )
    td_version, _ = fv.create_training_data(
        description="PM2.5 training dataset", write_options={"wait_for_job": True},
    )
    print("TRAINING DATASET version:", td_version)

    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
    X_train = X_train[FEATURES].astype("float64")
    X_test = X_test[FEATURES].astype("float64")
    y_train = y_train.values.ravel().astype("float64")
    y_test = y_test.values.ravel().astype("float64")

    # ================= TRAIN — pick best by CV =================
    candidates = {
        "linreg": LinearRegression(),
        "rf": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "gbr": GradientBoostingRegressor(random_state=42),
        "hgb": HistGradientBoostingRegressor(random_state=42),
    }
    best_name, best_cv = None, np.inf
    for name, mdl in candidates.items():
        scores = cross_val_score(mdl, X_train, y_train, cv=5,
                                 scoring="neg_root_mean_squared_error")
        cv_rmse = -scores.mean()
        print(f"CV {name}: rmse={cv_rmse:.4f}")
        if cv_rmse < best_cv:
            best_cv, best_name = cv_rmse, name

    model = candidates[best_name]
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))
    print(f"BEST={best_name} held-out RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

    # refit on all available history for best forecast accuracy
    X_all = pd.concat([X_train, X_test]); y_all = np.concatenate([y_train, y_test])
    model.fit(X_all, y_all)

    # ================= REGISTER MODEL =================
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    model_dir = "airq_model_dir"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    model_schema = ModelSchema(Schema(X_train), Schema(pd.DataFrame({"pm25": y_train})))
    hops_model = mr.sklearn.create_model(
        name="airqmodelf3a8d0",
        metrics={"rmse": rmse, "mae": mae, "r2": r2},
        model_schema=model_schema,
        input_example=X_train.head(),
        description=f"PM2.5 regressor ({best_name})",
    )
    hops_model.save(model_dir)
    print("MODEL registered:", hops_model.version)

    # ================= INFERENCE → PREDICTIONS FG =================
    Xf = fcst[FEATURES].astype("float64")
    fcst_out = pd.DataFrame({
        "date": fcst["date"].dt.strftime("%Y-%m-%d"),
        "pm25_pred": model.predict(Xf).astype("float64"),
    })
    pred_fg = fs.get_or_create_feature_group(
        name="airqpredf3a8d0", version=1,
        description="PM2.5 predictions for forecast days",
        primary_key=["date"], online_enabled=True,
    )
    pred_fg.insert(fcst_out, wait=True)
    print("PREDICTIONS inserted:", len(fcst_out))
    print("DONE")

if __name__ == "__main__":
    main()
