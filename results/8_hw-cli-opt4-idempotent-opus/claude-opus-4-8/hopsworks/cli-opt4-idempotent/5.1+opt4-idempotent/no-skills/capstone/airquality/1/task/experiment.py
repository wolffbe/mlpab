"""Backtest several models/feature sets to pick the best PM2.5 regressor (platform-side, read-only)."""
import numpy as np
import pandas as pd
import hopsworks
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error


def rmse(a, b):
    return float(mean_squared_error(a, b) ** 0.5)


def main():
    project = hopsworks.login()
    dsapi = project.get_dataset_api()
    hist = pd.read_csv(dsapi.download("Resources/airq/airquality_history.csv", overwrite=True))
    hist["_dt"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("_dt").reset_index(drop=True)
    hist["pm25_roll3"] = hist["pm25_lag1"].rolling(3, min_periods=1).mean()
    hist["pm25_roll7"] = hist["pm25_lag1"].rolling(7, min_periods=1).mean()
    hist["dlag"] = hist["pm25_lag1"] - hist["pm25_roll3"]

    raw = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]
    raw_roll = raw + ["pm25_roll3", "pm25_roll7", "dlag"]
    y = hist["pm25"]
    n = len(hist)
    split = int(n * 0.8)

    def backtest(feats, model_fn):
        X = hist[feats]
        # last-20% holdout
        m = model_fn(); m.fit(X.iloc[:split], y.iloc[:split])
        r1 = rmse(y.iloc[split:], m.predict(X.iloc[split:]))
        # 5-fold expanding-window
        rs = []
        for frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
            s = int(n * frac)
            mm = model_fn(); mm.fit(X.iloc[:s], y.iloc[:s])
            rs.append(rmse(y.iloc[s:min(s + n // 10, n)], mm.predict(X.iloc[s:min(s + n // 10, n)])))
        return r1, float(np.mean(rs))

    models = {
        "persistence": None,
        "ridge": lambda: Ridge(alpha=1.0),
        "linreg": lambda: LinearRegression(),
        "gbr": lambda: GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42),
        "gbr_shallow": lambda: GradientBoostingRegressor(n_estimators=600, max_depth=2, learning_rate=0.03, subsample=0.8, random_state=42),
        "rf": lambda: RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "et": lambda: ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "histgbr": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=3, random_state=42),
    }
    # persistence baseline
    r1 = rmse(y.iloc[split:], hist["pm25_lag1"].iloc[split:])
    print(f"persistence(pm25_lag1)  holdout={r1:.4f}")
    for fset_name, feats in [("raw6", raw), ("raw+roll", raw_roll)]:
        for name, fn in models.items():
            if fn is None:
                continue
            h, cv = backtest(feats, fn)
            print(f"{fset_name:9s} {name:12s} holdout={h:.4f}  cv={cv:.4f}")


if __name__ == "__main__":
    main()
