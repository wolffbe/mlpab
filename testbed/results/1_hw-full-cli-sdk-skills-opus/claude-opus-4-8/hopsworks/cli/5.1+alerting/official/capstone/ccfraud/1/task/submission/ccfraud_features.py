"""Shared fraud feature engineering. Runs platform-side inside Hopsworks jobs.

Both the feature pipeline and the inference pipeline import this module and call
`engineer(train_df, score_df)` with the SAME inputs, so the features computed for
the labelled history and for the scoring slice are guaranteed identical (no
training/serving skew).
"""
import numpy as np
import pandas as pd

# The numeric model inputs (no identifiers, no event time -> no leakage).
FEATURES = [
    "amount",
    "amt_log",
    "hour",
    "day_of_week",
    "is_night",
    "cat_fraud_rate",
    "dist_from_home",
    "dist_from_prev",
    "time_since_prev_min",
    "speed_kmph",
    "amount_to_avg",
    "txn_count_1h",
]


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two (lat, lon) arrays."""
    R = 6371.0
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def engineer(train_df, score_df):
    """Return a DataFrame with engineered features for train+score rows.

    A `__src` column marks each row ('train'/'score'). Per-card aggregates
    (home location, mean amount) use both sets (no label needed); the category
    fraud rate is learned from the labelled train rows only.
    """
    train_df = train_df.copy()
    score_df = score_df.copy()
    train_df["__src"] = "train"
    score_df["__src"] = "score"
    if "is_fraud" not in score_df.columns:
        score_df["is_fraud"] = np.nan

    cols = ["transaction_id", "cc_num", "datetime", "amount",
            "merchant", "category", "lat", "long", "is_fraud", "__src"]
    df = pd.concat([train_df[cols], score_df[cols]], ignore_index=True)

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values(["cc_num", "datetime"]).reset_index(drop=True)

    # Per-card "home" location and typical spend (label-free aggregates).
    home = df.groupby("cc_num")[["lat", "long"]].transform("median")
    df["home_lat"] = home["lat"]
    df["home_long"] = home["long"]
    df["card_mean_amt"] = df.groupby("cc_num")["amount"].transform("mean")

    # Category fraud rate, learned from labelled rows only.
    tr = df[df["__src"] == "train"]
    cat_rate = tr.groupby("category")["is_fraud"].mean()
    global_rate = float(tr["is_fraud"].mean())
    df["cat_fraud_rate"] = df["category"].map(cat_rate).fillna(global_rate)

    # Time signals.
    df["hour"] = df["datetime"].dt.hour.astype("int64")
    df["day_of_week"] = df["datetime"].dt.dayofweek.astype("int64")
    df["is_night"] = (df["hour"] < 6).astype("int64")
    df["amt_log"] = np.log1p(df["amount"])

    # Geo: distance from the card's usual location.
    df["dist_from_home"] = haversine(df["lat"], df["long"], df["home_lat"], df["home_long"])

    # Causal per-card sequence features (only look at the previous transaction).
    g = df.groupby("cc_num")
    df["prev_lat"] = g["lat"].shift(1)
    df["prev_long"] = g["long"].shift(1)
    df["prev_dt"] = g["datetime"].shift(1)
    df["dist_from_prev"] = haversine(df["lat"], df["long"],
                                     df["prev_lat"], df["prev_long"]).fillna(0.0)
    dt_min = (df["datetime"] - df["prev_dt"]).dt.total_seconds() / 60.0
    df["time_since_prev_min"] = dt_min.fillna(100000.0)
    # Implied travel speed: huge values flag impossible geographic jumps.
    df["speed_kmph"] = df["dist_from_prev"] / ((df["time_since_prev_min"] / 60.0) + 0.01)
    df["amount_to_avg"] = df["amount"] / (df["card_mean_amt"] + 1e-6)

    # Transaction velocity: count of same-card txns in the trailing hour (causal).
    counts = []
    for _, sub in df.groupby("cc_num", sort=False):
        s = sub.set_index("datetime")["amount"]
        c = s.rolling("1h").count() - 1  # exclude the current row itself
        counts.append(pd.Series(c.values, index=sub.index))
    df["txn_count_1h"] = pd.concat(counts).sort_index().astype("float64")

    for col in FEATURES:
        df[col] = df[col].astype("float64") if col not in ("hour", "day_of_week", "is_night") else df[col]
    return df


def load_inputs(project):
    """Download both CSVs from HopsFS and return (train_df, score_df)."""
    import os
    ds = project.get_dataset_api()
    base = "Resources/ccdata"
    for fn in ("transactions.csv", "score_transactions.csv"):
        if os.path.exists(fn):
            os.remove(fn)
        ds.download(f"{base}/{fn}", local_path=".", overwrite=True)
    train_df = pd.read_csv("transactions.csv")
    score_df = pd.read_csv("score_transactions.csv")
    return train_df, score_df
