"""Full FTI fraud pipeline — runs as a Hopsworks PYTHON job on the cluster.

Feature pipeline  -> feature group  cctxn4b8521
Training pipeline -> feature view + training dataset cctd4b8521, model ccmodel4b8521
Inference pipeline-> predictions feature group ccpred4b8521 (online enabled)
"""

import os
import pickle

import numpy as np
import pandas as pd

import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
ds_api = proj.get_dataset_api()

# ---------------------------------------------------------------- load data
DATA_DIR = "/hopsfs/Resources/ccfraud"
train_raw = pd.read_csv(f"{DATA_DIR}/transactions.csv", parse_dates=["datetime"])
score_raw = pd.read_csv(f"{DATA_DIR}/score_transactions.csv", parse_dates=["datetime"])
print("train:", train_raw.shape, "score:", score_raw.shape, flush=True)

train_raw["is_fraud"] = train_raw["is_fraud"].astype(int)
score_raw["is_fraud"] = np.nan

df = pd.concat([train_raw, score_raw], ignore_index=True)
df = df.sort_values(["cc_num", "datetime"]).reset_index(drop=True)

# -------------------------------------------------------- feature engineering
# All history-based features use only PAST rows (point-in-time correct):
# score rows come after the labelled history, so they see the full train past.

g = df.groupby("cc_num", sort=False)

# time features
df["hour"] = df["datetime"].dt.hour.astype(float)
df["dayofweek"] = df["datetime"].dt.dayofweek.astype(float)
df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(float)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# amount features
df["log_amount"] = np.log1p(df["amount"])

# per-card expanding stats of amount (past rows only via shift)
amt_shift = g["amount"].shift(1)
df["_amt_shift"] = amt_shift
gs = df.groupby("cc_num", sort=False)["_amt_shift"]
df["card_amt_mean"] = gs.transform(lambda s: s.expanding().mean())
df["card_amt_std"] = gs.transform(lambda s: s.expanding().std())
df["card_amt_max"] = gs.transform(lambda s: s.expanding().max())
df["amt_over_mean"] = df["amount"] / (df["card_amt_mean"] + 1.0)
df["amt_z"] = (df["amount"] - df["card_amt_mean"]) / (df["card_amt_std"] + 1.0)
df["amt_over_max"] = df["amount"] / (df["card_amt_max"] + 1.0)

# velocity: time since previous txn on same card, txn counts in windows
prev_time = g["datetime"].shift(1)
df["secs_since_prev"] = (df["datetime"] - prev_time).dt.total_seconds()
df["secs_since_prev"] = df["secs_since_prev"].fillna(7 * 24 * 3600).clip(0, 30 * 24 * 3600)
df["log_secs_since_prev"] = np.log1p(df["secs_since_prev"])

def past_count(sub, window):
    # rolling count over a time window including current row, minus current
    c = sub.rolling(window, on="datetime")["amount"].count() - 1.0
    return c

cnt1h, cnt24h = [], []
for _, sub in df.groupby("cc_num", sort=False):
    cnt1h.append(past_count(sub, "1h"))
    cnt24h.append(past_count(sub, "24h"))
df["txn_cnt_1h"] = pd.concat(cnt1h).sort_index()
df["txn_cnt_24h"] = pd.concat(cnt24h).sort_index()

# geo: distance from card's historical mean location, and from previous txn
lat_shift = g["lat"].shift(1)
lon_shift = g["long"].shift(1)
df["_lat_shift"] = lat_shift
df["_lon_shift"] = lon_shift
df["card_lat_mean"] = df.groupby("cc_num", sort=False)["_lat_shift"].transform(
    lambda s: s.expanding().mean()
)
df["card_lon_mean"] = df.groupby("cc_num", sort=False)["_lon_shift"].transform(
    lambda s: s.expanding().mean()
)

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

df["dist_from_home_km"] = haversine_km(
    df["lat"], df["long"], df["card_lat_mean"], df["card_lon_mean"]
)
df["dist_from_prev_km"] = haversine_km(
    df["lat"], df["long"], df["_lat_shift"], df["_lon_shift"]
)
# impossible-travel speed (km/h) from previous transaction
df["speed_kmh"] = df["dist_from_prev_km"] / (df["secs_since_prev"] / 3600.0 + 0.01)

# per-card prior fraud history (labels of PAST transactions only)
fraud_filled = df["is_fraud"].fillna(0)
df["_fr"] = fraud_filled
cum_fraud = df.groupby("cc_num", sort=False)["_fr"].cumsum() - df["_fr"]
df["card_prior_fraud_cnt"] = cum_fraud
prior_cnt = df.groupby("cc_num", sort=False).cumcount()
df["card_prior_txn_cnt"] = prior_cnt.astype(float)
df["card_prior_fraud_rate"] = df["card_prior_fraud_cnt"] / (df["card_prior_txn_cnt"] + 5.0)

# category one-hot
for cat in sorted(df["category"].unique()):
    df["cat_" + cat] = (df["category"] == cat).astype(float)

# merchant frequency encoding (no labels involved)
merch_freq = df["merchant"].value_counts(normalize=True)
df["merchant_freq"] = df["merchant"].map(merch_freq).astype(float)

FEATURES = (
    [
        "amount", "log_amount", "hour", "dayofweek", "is_night", "hour_sin",
        "hour_cos", "card_amt_mean", "card_amt_std", "amt_over_mean", "amt_z",
        "amt_over_max", "log_secs_since_prev", "txn_cnt_1h", "txn_cnt_24h",
        "dist_from_home_km", "dist_from_prev_km", "speed_kmh",
        "card_prior_fraud_cnt", "card_prior_txn_cnt", "card_prior_fraud_rate",
        "merchant_freq", "lat", "long",
    ]
    + ["cat_" + c for c in sorted(df["category"].unique())]
)

df[FEATURES] = df[FEATURES].fillna(0.0).replace([np.inf, -np.inf], 0.0)

labelled = df[df["is_fraud"].notna()].copy()
scoring = df[df["is_fraud"].isna()].copy()
labelled["is_fraud"] = labelled["is_fraud"].astype(int)

# ------------------------------------------------- feature group cctxn4b8521
fg_cols = ["transaction_id", "cc_num", "datetime", "is_fraud"] + FEATURES
fg_df = labelled[fg_cols].copy()
fg_df.columns = [c.lower() for c in fg_df.columns]

fg = fs.get_or_create_feature_group(
    name="cctxn4b8521",
    version=1,
    description="Engineered credit-card fraud features (labelled history)",
    primary_key=["transaction_id"],
    event_time="datetime",
    online_enabled=False,
)
fg.insert(fg_df, write_options={"wait_for_job": True})
print("feature group cctxn4b8521 written:", fg_df.shape, flush=True)

# --------------------------------------- feature view + training dataset cctd4b8521
query = fg.select_all()
fv = fs.get_or_create_feature_view(
    name="cctd4b8521",
    version=1,
    description="Fraud training dataset view over cctxn4b8521",
    query=query,
    labels=["is_fraud"],
)
td_version, td_job = fv.create_training_data(
    description="cctd4b8521 fraud training dataset",
    write_options={"wait_for_job": True},
)
print("training dataset version:", td_version, flush=True)

X_all, y_all = fv.get_training_data(td_version)
print("training data:", X_all.shape, flush=True)

# ------------------------------------------------------------- train + evaluate
feat_lc = [c.lower() for c in FEATURES]
X_all = X_all.sort_values("datetime").reset_index(drop=True)
y_all = y_all.loc[X_all.index] if len(y_all) == len(X_all) else y_all
# re-align labels by re-reading via merge on transaction_id to be safe
merged = X_all.merge(
    fg_df[["transaction_id", "is_fraud"]], on="transaction_id", how="left"
)
merged = merged.sort_values("datetime").reset_index(drop=True)
y_ser = merged["is_fraud"].astype(int)
X_feat = merged[feat_lc].astype(float)

split = int(len(merged) * 0.8)
X_tr, X_va = X_feat.iloc[:split], X_feat.iloc[split:]
y_tr, y_va = y_ser.iloc[:split], y_ser.iloc[split:]

from sklearn.metrics import average_precision_score, roc_auc_score

def make_model():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=2,
            eval_metric="auc",
            scale_pos_weight=float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)),
            n_jobs=4,
            random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=6, random_state=42
        )

clf = make_model()
clf.fit(X_tr, y_tr)
p_va = clf.predict_proba(X_va)[:, 1]
auc = float(roc_auc_score(y_va, p_va))
ap = float(average_precision_score(y_va, p_va))
print("holdout ROC AUC:", auc, "AP:", ap, flush=True)

# retrain on the full labelled data for final scoring
clf_full = make_model()
clf_full.fit(X_feat, y_ser)

# ------------------------------------------------------ register ccmodel4b8521
model_dir = "ccmodel4b8521_dir"
os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
    pickle.dump(clf_full, f)
with open(os.path.join(model_dir, "features.txt"), "w") as f:
    f.write("\n".join(feat_lc))

mr = proj.get_model_registry()
mdl = mr.python.create_model(
    name="ccmodel4b8521",
    metrics={"roc_auc": auc, "average_precision": ap},
    description="Credit-card fraud classifier (gradient boosting) trained on cctd4b8521",
    input_example=X_feat.head(1),
)
mdl.save(model_dir)
print("model registered: ccmodel4b8521", flush=True)

# ------------------------------------------------------------- score + publish
score_feat = scoring[FEATURES].astype(float)
score_feat.columns = feat_lc
proba = clf_full.predict_proba(score_feat)[:, 1]
pred_df = pd.DataFrame(
    {
        "transaction_id": scoring["transaction_id"].values,
        "fraud_probability": np.clip(proba.astype(float), 0.0, 1.0),
    }
)
print("predictions:", pred_df.shape, pred_df["fraud_probability"].describe(), flush=True)

pred_fg = fs.get_or_create_feature_group(
    name="ccpred4b8521",
    version=1,
    description="Fraud probability predictions for score_transactions",
    primary_key=["transaction_id"],
    online_enabled=True,
)
pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print("predictions feature group ccpred4b8521 written (online enabled)", flush=True)
print("PIPELINE_DONE holdout_roc_auc=", auc, flush=True)
