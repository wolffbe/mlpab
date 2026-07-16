"""Full FTI pipeline for PM2.5 forecasting — runs entirely on the Hopsworks platform.

F: engineer features into feature group `airq963ee7`
T: assemble training dataset under feature view `airqtd963ee7`, train + register `airqmodel963ee7`
I: predict forecast_days into online+offline feature group `airqpred963ee7`
"""
import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import hopsworks

FG_NAME = "airq963ee7"
FV_NAME = "airqtd963ee7"
MODEL_NAME = "airqmodel963ee7"
PRED_FG_NAME = "airqpred963ee7"

BASE = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]


def engineer(df):
    """Per-row features computable for BOTH history and forecast rows."""
    df = df.copy()
    df["temp_humidity"] = df["temperature"] * df["humidity"]
    df["wind_humidity"] = df["wind_speed"] * df["humidity"]
    df["lag1_temp"] = df["pm25_lag1"] * df["temperature"]
    df["lag1_sq"] = df["pm25_lag1"] ** 2
    df["wind_precip"] = df["wind_speed"] * df["precipitation"]
    return df


FEATURES = BASE + ["temp_humidity", "wind_humidity", "lag1_temp", "lag1_sq", "wind_precip"]


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dapi = project.get_dataset_api()

    # ---- read inputs uploaded to HopsFS ----
    for remote, local in [
        ("Resources/airqjob/airquality_history.csv", "hist.csv"),
        ("Resources/airqjob/forecast_days.csv", "fc.csv"),
    ]:
        if os.path.exists(local):
            os.remove(local)
        dapi.download(remote, local_path=local, overwrite=True)

    hist = pd.read_csv("hist.csv")
    fc = pd.read_csv("fc.csv")
    print("history rows:", len(hist), "forecast rows:", len(fc))

    hist = engineer(hist)
    hist["date"] = pd.to_datetime(hist["date"])

    # ============ F: feature group ============
    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=1,
        description="Daily air-quality + weather features with lag signals for PM2.5 forecasting",
        primary_key=["date"],
        event_time="date",
        online_enabled=False,
    )
    fg_cols = ["date"] + FEATURES + ["pm25"]
    fg.insert(hist[fg_cols], write_options={"wait_for_job": True})
    print("inserted feature group", FG_NAME)

    # ============ T: feature view + training dataset ============
    fv = None
    try:
        fv = fs.get_feature_view(name=FV_NAME, version=1)
    except Exception:
        fv = None
    if fv is None:
        query = fg.select(FEATURES + ["pm25"])
        fs.create_feature_view(
            name=FV_NAME,
            version=1,
            description="Training view for PM2.5 regressor",
            labels=["pm25"],
            query=query,
        )
        fv = fs.get_feature_view(name=FV_NAME, version=1)
    assert fv is not None, "feature view could not be created/fetched"
    print("feature view ready:", FV_NAME)

    td_version, _ = fv.create_train_test_split(
        test_size=0.2, write_options={"wait_for_job": True}
    )
    X_train, X_test, y_train, y_test = fv.get_train_test_split(
        training_dataset_version=td_version
    )
    X_train = X_train[FEATURES]
    X_test = X_test[FEATURES]
    y_train = y_train["pm25"].astype(float)
    y_test = y_test["pm25"].astype(float)
    print("train/test sizes:", len(X_train), len(X_test), "td_version:", td_version)

    # ============ T: train + evaluate ============
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    params = dict(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    eval_model = XGBRegressor(**params)
    eval_model.fit(X_train, y_train)
    preds = eval_model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"HELDOUT RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

    # final model on all history for best forecast accuracy
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    model = XGBRegressor(**params)
    model.fit(X_all, y_all)

    # ============ register model ============
    model_dir = "airq_model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/model.pkl")
    json.dump(FEATURES, open(f"{model_dir}/feature_names.json", "w"))

    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    mr = project.get_model_registry()
    hw_model = mr.python.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        description="XGBoost PM2.5 daily regressor (lag + weather features)",
        input_example=X_train.head(1),
        feature_view=fv,
        training_dataset_version=td_version,
    )
    hw_model.save(model_dir)
    print("registered model", MODEL_NAME, metrics)

    # ============ I: predict forecast -> online+offline FG ============
    fc_e = engineer(fc)
    fc_pred = model.predict(fc_e[FEATURES])
    out = pd.DataFrame({
        "date": fc["date"].astype(str),
        "pm25_pred": fc_pred.astype(float),
    })
    print("forecast predictions:", len(out))

    pred_fg = fs.get_or_create_feature_group(
        name=PRED_FG_NAME,
        version=1,
        description="PM2.5 predictions for forecast days",
        primary_key=["date"],
        online_enabled=True,
    )
    pred_fg.insert(out, write_options={"wait_for_job": True})
    print("inserted predictions into", PRED_FG_NAME, "(online+offline)")
    print("DONE")


if __name__ == "__main__":
    main()
