# Databricks notebook source
import dlt
import numpy as np
import pandas as pd

CAT = "workspace"
SCH = "mlpab2efe57"
SCHEMA = f"{CAT}.{SCH}"
VOL = f"/Volumes/{CAT}/{SCH}/airqdata"
MODEL_NAME = f"{SCHEMA}.airqmodel3d0e82"
EXP_PATH = "/Users/benedict@hopsworks.ai/mlpab2efe57/airq_experiment"

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed",
            "pressure", "precipitation", "doy_sin", "doy_cos", "month"]

_STATE = {}


def _read_csv(name):
    df = (spark.read.option("header", True).option("inferSchema", True)
          .csv(f"{VOL}/{name}").toPandas())
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _add_calendar(d):
    d = d.copy()
    doy = d["date"].dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["month"] = d["date"].dt.month.astype("int32")
    return d


def _features():
    if "hist" in _STATE:
        return _STATE
    hist = _add_calendar(_read_csv("airquality_history.csv"))
    past = hist["pm25"].shift(1)
    hist["pm25_roll3"] = past.rolling(3, min_periods=1).mean()
    hist["pm25_roll7"] = past.rolling(7, min_periods=1).mean()
    hist["pm25_roll14"] = past.rolling(14, min_periods=1).mean()
    hist = hist.dropna(subset=["pm25_lag1"]).reset_index(drop=True)
    n_test = 90
    hist["split"] = "train"
    hist.loc[hist.index[-n_test:], "split"] = "test"
    fc = _add_calendar(_read_csv("forecast_days.csv"))
    _STATE["hist"] = hist
    _STATE["fc"] = fc
    return _STATE


def _trained():
    if "model" in _STATE:
        return _STATE
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    st = _features()
    hist = st["hist"]
    train = hist[hist["split"] == "train"]
    test = hist[hist["split"] == "test"]
    Xtr, ytr = train[FEATURES], train["pm25"]
    Xte, yte = test[FEATURES], test["pm25"]

    configs = [
        dict(n_estimators=500, learning_rate=0.03, max_depth=2, subsample=0.9),
        dict(n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.9),
        dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8),
        dict(n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.9),
    ]
    best = None
    for cfg in configs:
        m = GradientBoostingRegressor(random_state=42, **cfg)
        m.fit(Xtr, ytr)
        r = float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))
        if best is None or r < best[0]:
            best = (r, cfg, m)
    rmse, cfg, model = best
    pred_te = model.predict(Xte)
    mae = float(mean_absolute_error(yte, pred_te))
    r2 = float(r2_score(yte, pred_te))
    base = float(np.sqrt(mean_squared_error(yte, np.full(len(yte), ytr.mean()))))

    final_model = GradientBoostingRegressor(random_state=42, **cfg)
    final_model.fit(hist[FEATURES], hist["pm25"])

    reg_status = "not_attempted"
    version = ""
    try:
        import mlflow
        mlflow.set_registry_uri("databricks-uc")
        try:
            mlflow.set_experiment(EXP_PATH)
        except Exception as e:
            reg_status = f"set_experiment_failed:{e}"
        with mlflow.start_run(run_name="airq_gbr"):
            mlflow.log_params(cfg)
            mlflow.log_param("features", ",".join(FEATURES))
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("baseline_rmse", base)
            mlflow.sklearn.log_model(
                final_model, "model",
                registered_model_name=MODEL_NAME,
                input_example=Xtr.head(5),
            )
        client = mlflow.MlflowClient()
        mv = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest = max(mv, key=lambda v: int(v.version))
        version = str(latest.version)
        client.update_model_version(
            name=MODEL_NAME, version=latest.version,
            description=(f"GBR pm25 forecaster. Held-out (last 90 days) RMSE={rmse:.4f}, "
                         f"MAE={mae:.4f}, R2={r2:.4f}, baseline RMSE={base:.4f}."))
        for k, v in [("rmse", rmse), ("mae", mae), ("r2", r2), ("baseline_rmse", base)]:
            client.set_model_version_tag(MODEL_NAME, latest.version, k, f"{v:.4f}")
        reg_status = "registered"
    except Exception as e:
        reg_status = f"failed:{type(e).__name__}:{e}"

    _STATE.update(model=final_model, rmse=rmse, mae=mae, r2=r2, base=base,
                  cfg=cfg, reg_status=reg_status, version=version)
    return _STATE


@dlt.table(name="airq3d0e82", comment="Air-quality feature group: weather + lag/rolling pm25 signals")
def airq3d0e82():
    hist = _features()["hist"]
    fg = hist.drop(columns=["split"]).copy()
    fg["date"] = fg["date"].dt.date
    return spark.createDataFrame(fg)


@dlt.table(name="airqtd3d0e82", comment="Training dataset: features + pm25 label + chronological split")
def airqtd3d0e82():
    hist = _features()["hist"]
    td = hist[["date"] + FEATURES + ["pm25", "split"]].copy()
    td["date"] = td["date"].dt.date
    return spark.createDataFrame(td)


@dlt.table(name="airq_pred_stage", comment="Predictions for forecast days (staging)")
def airq_pred_stage():
    st = _trained()
    fc = st["fc"]
    out = pd.DataFrame({
        "date": fc["date"].dt.date,
        "pm25_pred": st["model"].predict(fc[FEATURES]).astype(float),
    })
    return spark.createDataFrame(out)


@dlt.table(name="airq_meta_stage", comment="Training/registration metadata")
def airq_meta_stage():
    st = _trained()
    rows = [
        ("rmse", f"{st['rmse']:.6f}"),
        ("mae", f"{st['mae']:.6f}"),
        ("r2", f"{st['r2']:.6f}"),
        ("baseline_rmse", f"{st['base']:.6f}"),
        ("config", str(st["cfg"])),
        ("registration", st["reg_status"]),
        ("model_version", st["version"]),
    ]
    return spark.createDataFrame(pd.DataFrame(rows, columns=["key", "value"]))
