"""Full FTI pipeline for PM2.5 forecasting, executed as a Hopsworks job (platform-side).

Steps:
  1. Feature engineering -> feature group `airq9e9046`
  2. Training dataset    -> `airqtd9e9046`
  3. Train + register    -> model `airqmodel9e9046` (with metrics)
  4. Batch inference     -> predictions feature group `airqpred9e9046` (online+offline)
"""
import os
import math
import traceback

import numpy as np
import pandas as pd

import hopsworks

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]
LABEL = "pm25"

HIST_REMOTE = "Resources/airq/airquality_history.csv"
FCAST_REMOTE = "Resources/airq/forecast_days.csv"


def _download(dataset_api, remote):
    """Download a HopsFS file to the job's local cwd and return the local path."""
    try:
        local = dataset_api.download(remote, overwrite=True)
    except TypeError:
        local = dataset_api.download(remote)
    if local and os.path.exists(local):
        return local
    base = os.path.basename(remote)
    if os.path.exists(base):
        return base
    raise FileNotFoundError(f"could not download {remote}")


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    # ---- Load raw data (uploaded to HopsFS) ---------------------------------
    hist_local = _download(dataset_api, HIST_REMOTE)
    fcast_local = _download(dataset_api, FCAST_REMOTE)
    hist = pd.read_csv(hist_local)
    fcast = pd.read_csv(fcast_local)
    print("history shape:", hist.shape, "forecast shape:", fcast.shape)

    # ---- 1. Feature engineering --------------------------------------------
    hist["date"] = hist["date"].astype(str)
    hist["event_time"] = pd.to_datetime(hist["date"])
    for c in FEATURES + [LABEL]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    hist = hist.dropna(subset=FEATURES + [LABEL]).reset_index(drop=True)

    fg_cols = ["date", "event_time"] + FEATURES + [LABEL]
    fg_df = hist[fg_cols].copy()

    fg = fs.get_or_create_feature_group(
        name="airq9e9046",
        version=1,
        description="Air-quality PM2.5 daily features (weather + lag1)",
        primary_key=["date"],
        event_time="event_time",
        online_enabled=False,
    )
    fg.insert(fg_df, write_options={"wait_for_job": True})
    print("inserted feature group airq9e9046:", fg_df.shape)

    # ---- 2. Training dataset airqtd9e9046 ----------------------------------
    td_df = hist[FEATURES + [LABEL]].copy()
    try:
        td = fs.create_training_dataset(
            name="airqtd9e9046",
            version=1,
            description="PM2.5 training dataset (80/20 split)",
            data_format="csv",
            label=[LABEL],
            splits={"train": 0.8, "test": 0.2},
            seed=42,
        )
        td.save(td_df)
        print("created legacy training dataset airqtd9e9046")
    except Exception as e:  # noqa: BLE001
        print("WARNING: legacy create_training_dataset failed:", repr(e))
        traceback.print_exc()
        # Fall back to a feature-view-based training dataset under the same name.
        query = fg.select(FEATURES + [LABEL])
        fv = fs.get_or_create_feature_view(
            name="airqtd9e9046", version=1, query=query, labels=[LABEL],
        )
        fv.create_train_test_split(test_size=0.2, description="PM2.5 td", seed=42)
        print("created feature-view-based training dataset airqtd9e9046")

    # ---- 3. Train + register model airqmodel9e9046 -------------------------
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    X = td_df[FEATURES]
    y = td_df[LABEL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3,
        subsample=0.9, random_state=42,
    )
    model.fit(X_tr, y_tr)
    pred_te = model.predict(X_te)
    rmse = float(math.sqrt(mean_squared_error(y_te, pred_te)))
    mae = float(mean_absolute_error(y_te, pred_te))
    r2 = float(r2_score(y_te, pred_te))
    print(f"held-out test RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

    # Refit on ALL history for the best forecast predictions.
    model.fit(X, y)

    import joblib
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema

    model_dir = "airq_model_dir"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    input_schema = Schema(X)
    output_schema = Schema(y)
    model_schema = ModelSchema(input_schema, output_schema)

    mr = project.get_model_registry()
    hops_model = mr.sklearn.create_model(
        name="airqmodel9e9046",
        metrics={"rmse": rmse, "mae": mae, "r2": r2},
        description="PM2.5 daily regressor (GradientBoosting)",
        input_example=X.iloc[:1],
        model_schema=model_schema,
    )
    hops_model.save(model_dir)
    print("registered model airqmodel9e9046 with metrics:", {"rmse": rmse, "mae": mae, "r2": r2})

    # ---- 4. Batch inference -> predictions FG airqpred9e9046 ---------------
    fcast["date"] = fcast["date"].astype(str)
    for c in FEATURES:
        fcast[c] = pd.to_numeric(fcast[c], errors="coerce")
    Xf = fcast[FEATURES].copy()
    preds = model.predict(Xf)

    pred_df = pd.DataFrame({
        "date": fcast["date"].values,
        "pm25_pred": np.asarray(preds, dtype="float64"),
    })

    pred_fg = fs.get_or_create_feature_group(
        name="airqpred9e9046",
        version=1,
        description="PM2.5 predictions for forecast days",
        primary_key=["date"],
        online_enabled=True,
    )
    pred_fg.insert(pred_df, write_options={"wait_for_job": True})
    print("inserted predictions feature group airqpred9e9046:", pred_df.shape)
    print(pred_df.head(10).to_string())
    print("DONE")


if __name__ == "__main__":
    main()
