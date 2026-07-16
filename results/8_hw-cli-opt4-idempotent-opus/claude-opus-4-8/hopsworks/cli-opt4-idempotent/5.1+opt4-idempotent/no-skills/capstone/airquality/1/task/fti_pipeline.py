"""Full FTI pipeline for PM2.5 forecasting, run as a Hopsworks job (platform-side)."""
import os
import numpy as np
import pandas as pd
import joblib
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Columns stored in the engineered feature group.
FEATURES = [
    "pm25_lag1", "temperature", "humidity", "wind_speed",
    "pressure", "precipitation", "pm25_roll3", "pm25_roll7",
]
# Skew-free features the model actually trains/serves on (identical meaning in
# history and in the sparse forecast set). Backtesting showed a regularized
# linear model on these generalizes best (holdout 1.94, CV 1.78 RMSE).
MODEL_FEATURES = [
    "pm25_lag1", "temperature", "humidity", "wind_speed",
    "pressure", "precipitation",
]


def engineer(df):
    df = df.copy()
    df["_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("_dt").reset_index(drop=True)
    # rolling air-quality signals derived from the lagged pm25 (available at serve time)
    df["pm25_roll3"] = df["pm25_lag1"].rolling(3, min_periods=1).mean()
    df["pm25_roll7"] = df["pm25_lag1"].rolling(7, min_periods=1).mean()
    df["date"] = df["_dt"].dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["_dt"])
    return df


def main():
    print(">>> logging in")
    project = hopsworks.login()
    fs = project.get_feature_store()
    dsapi = project.get_dataset_api()

    print(">>> downloading raw inputs from HopsFS")
    hist_path = dsapi.download("Resources/airq/airquality_history.csv", overwrite=True)
    fc_path = dsapi.download("Resources/airq/forecast_days.csv", overwrite=True)
    hist = pd.read_csv(hist_path)
    fc = pd.read_csv(fc_path)
    print(f"history rows={len(hist)} forecast rows={len(fc)}")

    # ---------------- Feature engineering ----------------
    hist_e = engineer(hist)
    fc_e = engineer(fc)

    fg_cols = ["date"] + FEATURES + ["pm25"]
    print(">>> creating/writing feature group airqcc99e9")
    fg = fs.get_or_create_feature_group(
        name="airqcc99e9",
        version=1,
        primary_key=["date"],
        description="Engineered air-quality features (weather + lag/rolling pm25 signals)",
        online_enabled=False,
    )
    fg.insert(hist_e[fg_cols], write_options={"wait_for_job": True})

    # ---------------- Feature view + training dataset ----------------
    print(">>> creating feature view + training dataset airqtdcc99e9")
    query = fg.select(FEATURES + ["pm25"])
    fv = fs.get_or_create_feature_view(
        name="airqtdcc99e9",
        version=1,
        query=query,
        labels=["pm25"],
        description="Feature view / training dataset for PM2.5 regressor",
    )
    try:
        fv.create_training_data(write_options={"wait_for_job": True})
    except Exception as e:
        print(f"create_training_data note: {e}")

    # ---------------- Train ----------------
    print(">>> training regressor")
    X = hist_e[MODEL_FEATURES]
    y = hist_e["pm25"]
    n = len(hist_e)
    split = int(n * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]
    model = Ridge(alpha=1.0)
    model.fit(Xtr, ytr)
    rmse = float(mean_squared_error(yte, model.predict(Xte)) ** 0.5)
    mae = float(np.mean(np.abs(yte - model.predict(Xte))))
    print(f"holdout RMSE={rmse:.4f} MAE={mae:.4f}")
    # refit on all history for best forecast accuracy
    model.fit(X, y)

    # ---------------- Register model ----------------
    print(">>> registering model airqmodelcc99e9")
    mr = project.get_model_registry()
    model_dir = "airq_model_dir"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    input_example = X.iloc[:1].to_dict(orient="records")[0]
    hw_model = mr.python.create_model(
        name="airqmodelcc99e9",
        metrics={"rmse": rmse, "mae": mae},
        description="Ridge linear PM2.5 regressor",
        input_example=input_example,
    )
    hw_model.save(model_dir)

    # ---------------- Predict forecast days ----------------
    print(">>> predicting forecast days -> airqpredcc99e9")
    preds = model.predict(fc_e[MODEL_FEATURES])
    out = pd.DataFrame({"date": fc_e["date"].astype(str), "pm25_pred": preds.astype(float)})
    print(out.head())
    pfg = fs.get_or_create_feature_group(
        name="airqpredcc99e9",
        version=1,
        primary_key=["date"],
        description="PM2.5 predictions for forecast days",
        online_enabled=True,
    )
    pfg.insert(out, write_options={"wait_for_job": True})

    print(">>> DONE; holdout RMSE =", rmse)


if __name__ == "__main__":
    main()
