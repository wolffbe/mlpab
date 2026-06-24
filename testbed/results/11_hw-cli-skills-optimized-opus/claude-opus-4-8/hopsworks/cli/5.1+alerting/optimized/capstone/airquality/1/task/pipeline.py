"""Full FTI air-quality PM2.5 pipeline, runs ON the Hopsworks platform as a job."""
import os
import math
import numpy as np
import pandas as pd
import joblib
import hopsworks
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

FG_NAME = "airqed0d86"
FV_NAME = "airqtded0d86"
MODEL_NAME = "airqmodeled0d86"
PRED_FG_NAME = "airqpreded0d86"

FEATURES = [
    "pm25_lag1", "temperature", "humidity", "wind_speed", "pressure",
    "precipitation", "month", "dayofyear", "doy_sin", "doy_cos",
]


def engineer(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month.astype("int64")
    df["dayofyear"] = df["date"].dt.dayofyear.astype("int64")
    doy = df["dayofyear"].astype(float)
    df["doy_sin"] = np.sin(2 * math.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * math.pi * doy / 365.25)
    for c in ["pm25_lag1", "temperature", "humidity", "wind_speed",
              "pressure", "precipitation", "doy_sin", "doy_cos"]:
        df[c] = df[c].astype(float)
    return df


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    ds = project.get_dataset_api()

    # ---- download inputs from HopsFS ----
    for fn in ["airquality_history.csv", "forecast_days.csv"]:
        if os.path.exists(fn):
            os.remove(fn)
        ds.download(f"Resources/airq/{fn}", local_path=fn, overwrite=True)

    hist = engineer(pd.read_csv("airquality_history.csv"))
    hist["pm25"] = hist["pm25"].astype(float)

    # ================= FEATURE PIPELINE =================
    # recreate FG fresh so schema matches our engineered dtypes
    try:
        old = fs.get_feature_group(FG_NAME, version=1)
        if old is not None:
            old.delete()
            print("deleted existing FG")
    except Exception as e:
        print("no existing FG:", e)

    fg = fs.create_feature_group(
        name=FG_NAME, version=1,
        primary_key=["date"], event_time="date",
        online_enabled=False,
        description="Air quality daily features: weather + pm25 lag + seasonality; target pm25",
    )
    fg_cols = ["date"] + FEATURES + ["pm25"]
    fg.insert(hist[fg_cols])
    print("FG inserted rows:", len(hist))

    # ================= TRAINING DATASET =================
    fv = fs.get_or_create_feature_view(
        name=FV_NAME, version=1,
        query=fg.select_all(),
        labels=["pm25"],
        description="Feature view for PM2.5 regression",
    )
    X_train, X_test, y_train, y_test = fv.train_test_split(
        test_size=0.2, description="train/test split for PM2.5 model"
    )
    print("td versions:", [td["version"] if isinstance(td, dict) else td
                           for td in fv.get_training_datasets()] if hasattr(fv, "get_training_datasets") else "n/a")

    Xtr = X_train[FEATURES].astype(float)
    Xte = X_test[FEATURES].astype(float)
    ytr = np.ravel(y_train.values.astype(float))
    yte = np.ravel(y_test.values.astype(float))

    # ================= TRAIN =================
    model = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_depth=4,
        max_leaf_nodes=31, l2_regularization=1.0,
        min_samples_leaf=20, random_state=42,
    )
    model.fit(Xtr, ytr)
    pred_te = model.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(yte, pred_te)))
    mae = float(np.mean(np.abs(yte - pred_te)))
    print(f"HELD-OUT RMSE={rmse:.4f} MAE={mae:.4f}")

    # ================= REGISTER MODEL =================
    mr = project.get_model_registry()
    art_dir = "airq_model"
    os.makedirs(art_dir, exist_ok=True)
    joblib.dump(model, os.path.join(art_dir, "model.pkl"))
    input_example = Xtr.head(1).to_dict(orient="records")[0]
    m = mr.python.create_model(
        name=MODEL_NAME,
        metrics={"rmse": rmse, "mae": mae},
        description="PM2.5 daily regressor (HistGradientBoosting)",
        input_example=input_example,
        feature_view=fv,
    )
    m.save(art_dir)
    print("registered model", MODEL_NAME, "with rmse", rmse)

    # ================= INFERENCE on forecast days =================
    fc = engineer(pd.read_csv("forecast_days.csv"))
    Xf = fc[FEATURES].astype(float)
    fc_pred = model.predict(Xf)
    out = pd.DataFrame({
        "date": pd.to_datetime(fc["date"]),
        "pm25_pred": fc_pred.astype(float),
    })

    try:
        oldp = fs.get_feature_group(PRED_FG_NAME, version=1)
        if oldp is not None:
            oldp.delete()
            print("deleted existing pred FG")
    except Exception as e:
        print("no existing pred FG:", e)

    pred_fg = fs.create_feature_group(
        name=PRED_FG_NAME, version=1,
        primary_key=["date"], event_time="date",
        online_enabled=True,
        description="PM2.5 predictions for forecast days (online+offline)",
    )
    pred_fg.insert(out)
    print("predictions inserted:", len(out))
    print("DONE")


if __name__ == "__main__":
    main()
