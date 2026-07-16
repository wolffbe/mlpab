"""Model-independent feature engineering for credit-card fraud detection.
Pure pandas/numpy. Used both to build the training feature group and the
score feature group, so train/serve features are computed identically.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "amount", "log_amount", "hour", "is_night",
    "time_since_prev_s", "dist_prev_km", "speed_kmh",
    "amt_z", "dist_home_km", "count_1h", "count_24h",
    "cat_fraud_rate",
]


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _rolling_counts(df):
    """Prior-transaction counts per card within 1h / 24h windows (exclude self)."""
    c1 = np.zeros(len(df))
    c24 = np.zeros(len(df))
    pos = {c: i for i, c in enumerate(df.columns)}
    # work per card on the time-sorted frame
    for _, idx in df.groupby("cc_num").groups.items():
        sub = df.loc[idx]
        t = sub["datetime"].values.astype("datetime64[s]").astype("int64")
        n = len(t)
        j1 = 0
        j24 = 0
        out1 = np.empty(n)
        out24 = np.empty(n)
        for i in range(n):
            while t[i] - t[j1] > 3600:
                j1 += 1
            while t[i] - t[j24] > 86400:
                j24 += 1
            out1[i] = i - j1
            out24[i] = i - j24
        c1[df.index.get_indexer(idx)] = out1
        c24[df.index.get_indexer(idx)] = out24
    return c1, c24


def engineer(df, fraud_rate_map, global_rate):
    """df has columns: transaction_id, cc_num, datetime, amount, merchant,
    category, lat, long [, is_fraud]. Returns df sorted with FEATURES added.
    Lookback features use only PRIOR rows of the same card (shift), so no
    same-row leakage; pass a df whose history precedes the rows you score.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values(["cc_num", "datetime"]).reset_index(drop=True)
    g = df.groupby("cc_num")

    df["log_amount"] = np.log1p(df["amount"].astype(float))
    df["hour"] = df["datetime"].dt.hour.astype(float)
    df["is_night"] = (df["hour"] < 6).astype(float)

    prev_dt = g["datetime"].shift(1)
    df["time_since_prev_s"] = (df["datetime"] - prev_dt).dt.total_seconds()

    prev_lat = g["lat"].shift(1)
    prev_long = g["long"].shift(1)
    df["dist_prev_km"] = _haversine(df["lat"].values, df["long"].values,
                                    prev_lat.values, prev_long.values)
    hrs = df["time_since_prev_s"] / 3600.0
    df["speed_kmh"] = np.where(hrs > 0, df["dist_prev_km"] / hrs, 0.0)

    amt_mean_prev = g["amount"].apply(
        lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    amt_std_prev = g["amount"].apply(
        lambda s: s.shift(1).expanding().std()).reset_index(level=0, drop=True)
    df["amt_z"] = (df["amount"] - amt_mean_prev) / amt_std_prev

    lat_mean_prev = g["lat"].apply(
        lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    long_mean_prev = g["long"].apply(
        lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    df["dist_home_km"] = _haversine(df["lat"].values, df["long"].values,
                                    lat_mean_prev.values, long_mean_prev.values)

    c1, c24 = _rolling_counts(df)
    df["count_1h"] = c1
    df["count_24h"] = c24

    df["cat_fraud_rate"] = df["category"].map(fraud_rate_map).fillna(global_rate)

    # clean up: first-of-card lookbacks are NaN -> neutral fills
    df["time_since_prev_s"] = df["time_since_prev_s"].fillna(7 * 86400.0)
    df["dist_prev_km"] = df["dist_prev_km"].fillna(0.0)
    df["speed_kmh"] = df["speed_kmh"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["amt_z"] = df["amt_z"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["dist_home_km"] = df["dist_home_km"].fillna(0.0)

    for c in FEATURES:
        df[c] = df[c].astype(float)
    return df
