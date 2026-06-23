"""
Orchestrator — uses only hopsworks SDK + pandas/numpy for data prep.
ML training/scoring run as Hopsworks platform jobs.
"""
import os
import time
import hopsworks
import pandas as pd
import numpy as np

# ── Connect ──────────────────────────────────────────────────────────────────
print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# ── 1. Load raw data ─────────────────────────────────────────────────────────
txn = pd.read_csv("data/transactions.csv")
score = pd.read_csv("data/score_transactions.csv")
print(f"Training rows: {len(txn)}, Score rows: {len(score)}")

# ── 2. Feature engineering ────────────────────────────────────────────────────

def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def engineer_features(df, card_profiles=None, cat_map=None):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["cc_num", "datetime"]).reset_index(drop=True)

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    if cat_map is None:
        cat_map = {c: i for i, c in enumerate(sorted(df["category"].unique()))}
    df["category_enc"] = df["category"].map(cat_map).fillna(-1).astype(int)

    df["ts"] = df["datetime"].astype(np.int64) // 1_000_000_000

    v1h_d, v24h_d, amt24_d = {}, {}, {}
    for cc, grp in df.groupby("cc_num", sort=False):
        ts_arr = grp["ts"].values
        amt_arr = grp["amount"].values
        idx_arr = grp.index.values
        for i in range(len(ts_arr)):
            t = ts_arr[i]
            prev_ts = ts_arr[:i]
            v1h_d[idx_arr[i]] = int(np.sum(prev_ts >= t - 3600))
            mask24 = prev_ts >= t - 86400
            v24h_d[idx_arr[i]] = int(np.sum(mask24))
            amt24_d[idx_arr[i]] = float(np.mean(amt_arr[:i][mask24])) if mask24.any() else float(amt_arr[i])

    df["velocity_1h"] = pd.Series(v1h_d)
    df["velocity_24h"] = pd.Series(v24h_d)
    df["amt_mean_24h"] = pd.Series(amt24_d)
    df["amt_ratio_24h"] = df["amount"] / (df["amt_mean_24h"] + 1e-3)

    if card_profiles is None:
        card_profiles = df.groupby("cc_num").agg(
            home_lat=("lat", "mean"),
            home_long=("long", "mean"),
            avg_amount=("amount", "mean"),
        ).reset_index()

    df = df.merge(card_profiles, on="cc_num", how="left")
    df["geo_distance"] = haversine_vec(
        df["lat"].values, df["long"].values,
        df["home_lat"].values, df["home_long"].values,
    )
    df["amount_zscore"] = (df["amount"] - df["avg_amount"]) / (df["avg_amount"] + 1e-3)
    return df, card_profiles, cat_map

print("Engineering training features...")
txn_feat, card_profiles, cat_map = engineer_features(txn)

# ── 3. Upload training features ───────────────────────────────────────────────
TXN_COLS = [
    "transaction_id", "cc_num", "hour", "day_of_week", "is_weekend", "is_night",
    "category_enc", "amount", "velocity_1h", "velocity_24h",
    "amt_mean_24h", "amt_ratio_24h", "geo_distance", "amount_zscore",
    "lat", "long", "is_fraud",
]
fg_df = txn_feat[TXN_COLS].copy()
fg_df["transaction_id"] = fg_df["transaction_id"].astype(str)
fg_df["cc_num"] = fg_df["cc_num"].astype(str)
for c in ["velocity_1h", "velocity_24h", "is_weekend", "is_night", "category_enc", "is_fraud"]:
    fg_df[c] = fg_df[c].astype(int)

print(f"Uploading {len(fg_df)} rows to cctxn89f322...")
fg = fs.get_or_create_feature_group(
    name="cctxn89f322",
    version=1,
    primary_key=["transaction_id"],
    description="CC fraud features",
    online_enabled=False,
)
fg.insert(fg_df, write_options={"wait_for_job": True})
print("cctxn89f322 insert done.")

# ── 4. Upload score features ───────────────────────────────────────────────────
print("Engineering score features...")
score_feat, _, _ = engineer_features(score, card_profiles=card_profiles, cat_map=cat_map)
g_lat = float(txn_feat["lat"].mean())
g_lon = float(txn_feat["long"].mean())
g_amt = float(txn_feat["amount"].mean())
for col, val in [("home_lat", g_lat), ("home_long", g_lon), ("avg_amount", g_amt), ("geo_distance", 0.0), ("amount_zscore", 0.0)]:
    score_feat[col] = score_feat[col].fillna(val)

SCORE_COLS = [c for c in TXN_COLS if c != "is_fraud"]
sfg_df = score_feat[SCORE_COLS].copy()
sfg_df["transaction_id"] = sfg_df["transaction_id"].astype(str)
sfg_df["cc_num"] = sfg_df["cc_num"].astype(str)
for c in ["velocity_1h", "velocity_24h", "is_weekend", "is_night", "category_enc"]:
    sfg_df[c] = sfg_df[c].astype(int)

print(f"Uploading {len(sfg_df)} rows to cctxnscore89f322...")
fg_score = fs.get_or_create_feature_group(
    name="cctxnscore89f322",
    version=1,
    primary_key=["transaction_id"],
    description="CC fraud score features",
    online_enabled=False,
)
fg_score.insert(sfg_df, write_options={"wait_for_job": True})
print("cctxnscore89f322 insert done.")

# ── 5. Create feature view + training dataset ─────────────────────────────────
fv = fs.get_feature_view("cctd89f322", version=1)
if fv is None:
    fv = fs.create_feature_view(
        name="cctd89f322",
        version=1,
        query=fg.select_all(),
        labels=["is_fraud"],
    )
    print("Feature view cctd89f322 created.")
else:
    print("Feature view cctd89f322 already exists.")

print("Creating train-test split dataset...")
td_version, td_job = fv.create_train_test_split(
    test_size=0.2,
    description="ccfraud training dataset",
    data_format="csv",
    write_options={"wait_for_job": True},
)
print(f"Training dataset version: {td_version}")

# ── 6. Submit training job ─────────────────────────────────────────────────────
print("Submitting training job to platform...")
jobs_api = project.get_jobs_api()
dataset_api = project.get_dataset_api()

dataset_api.upload("train_job.py", "Resources", overwrite=True)

train_job = jobs_api.get_job("cctrain89f322")
if train_job is None:
    cfg = jobs_api.get_configuration("PYTHON")
    cfg["appPath"] = "Resources/train_job.py"
    cfg["defaultArgs"] = str(td_version)
    train_job = jobs_api.create_job("cctrain89f322", cfg)
    print("Training job created.")
else:
    print("Training job already exists.")

print(f"Running training job with td_version={td_version}...")
train_exec = train_job.run(args=str(td_version), await_termination=True)
print(f"Training job state: {train_exec.state}")

# ── 7. Submit scoring job ──────────────────────────────────────────────────────
print("Submitting scoring job to platform...")
dataset_api.upload("score_job.py", "Resources", overwrite=True)

score_job = jobs_api.get_job("ccscore89f322")
if score_job is None:
    cfg2 = jobs_api.get_configuration("PYTHON")
    cfg2["appPath"] = "Resources/score_job.py"
    score_job = jobs_api.create_job("ccscore89f322", cfg2)
    print("Scoring job created.")
else:
    print("Scoring job already exists.")

print("Running scoring job...")
score_exec = score_job.run(await_termination=True)
print(f"Scoring job state: {score_exec.state}")

print("\n=== PIPELINE COMPLETE ===")
print(f"  cctxn89f322 v1: {len(fg_df)} rows")
print(f"  cctxnscore89f322 v1: {len(sfg_df)} rows")
print(f"  cctd89f322 v1 (training dataset v{td_version})")
print(f"  ccmodel89f322 v1")
print(f"  ccpred89f322 v1")
