import os
import sys
import warnings
warnings.filterwarnings("ignore")

import hopsworks
import pandas as pd
import numpy as np

print("Logging in to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# ── Load raw data ──────────────────────────────────────────────────────────────
dataset_api = project.get_dataset_api()

print("Downloading data files...")
dataset_api.download("Resources/transactions.csv", local_path="/tmp/transactions.csv", overwrite=True)
dataset_api.download("Resources/score_transactions.csv", local_path="/tmp/score_transactions.csv", overwrite=True)

train_df = pd.read_csv("/tmp/transactions.csv", parse_dates=["datetime"])
score_df = pd.read_csv("/tmp/score_transactions.csv", parse_dates=["datetime"])
print(f"Train rows: {len(train_df)}, Score rows: {len(score_df)}")

# ── Feature engineering ────────────────────────────────────────────────────────

def compute_velocity(df):
    df_sorted = df.sort_values(["cc_num", "ts_unix"]).reset_index(drop=True)
    velocities = np.zeros(len(df_sorted), dtype="int64")
    for cc, grp in df_sorted.groupby("cc_num", sort=False):
        idx = grp.index.values
        ts_vals = grp["ts_unix"].values
        for i, row_idx in enumerate(idx):
            cutoff = ts_vals[i] - 86400
            velocities[row_idx] = int(np.sum(ts_vals[:i] >= cutoff))
    return df_sorted["ts_unix"].copy().rename("_dummy").map(
        lambda x: x  # just to get the right index
    ).pipe(lambda _: pd.Series(velocities, index=df_sorted.index))

def engineer_features(df, card_stats=None):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["hour"] = df["datetime"].dt.hour.astype("int64")
    df["day_of_week"] = df["datetime"].dt.dayofweek.astype("int64")
    df["ts_unix"] = (df["datetime"].astype("int64") // 10**9).astype("int64")

    is_train = card_stats is None
    if is_train:
        card_stats = df.groupby("cc_num").agg(
            card_mean_amount=("amount", "mean"),
            card_std_amount=("amount", lambda x: float(x.std()) if len(x) > 1 else 0.0),
            card_tx_count=("transaction_id", "count"),
            card_mean_lat=("lat", "mean"),
            card_mean_long=("long", "mean"),
        ).reset_index()

    df = df.merge(card_stats, on="cc_num", how="left")
    df["card_mean_amount"] = df["card_mean_amount"].fillna(df["amount"])
    df["card_std_amount"] = df["card_std_amount"].fillna(0.0)
    df["card_tx_count"] = df["card_tx_count"].fillna(1.0).astype("int64")
    df["card_mean_lat"] = df["card_mean_lat"].fillna(df["lat"])
    df["card_mean_long"] = df["card_mean_long"].fillna(df["long"])

    df["amount_deviation"] = (df["amount"] - df["card_mean_amount"]) / (df["card_std_amount"] + 1.0)
    df["log_amount"] = np.log1p(df["amount"])
    df["is_round_amount"] = ((df["amount"] % 10) < 0.01).astype("int64")
    dlat = df["lat"] - df["card_mean_lat"]
    dlong = df["long"] - df["card_mean_long"]
    df["geo_distance"] = np.sqrt(dlat**2 + dlong**2) * 111.0

    # Velocity
    vel = compute_velocity(df)
    df["tx_velocity_24h"] = vel.values.astype("int64")

    df["category_code"] = df["category"].astype("category").cat.codes.astype("int64")
    df["amount_hour_mean"] = df.groupby("hour")["amount"].transform("mean")

    if is_train:
        return df, card_stats
    return df

FEATURE_COLS = [
    "transaction_id", "cc_num", "amount", "hour", "day_of_week",
    "card_mean_amount", "card_std_amount", "card_tx_count",
    "card_mean_lat", "card_mean_long",
    "amount_deviation", "geo_distance", "tx_velocity_24h",
    "amount_hour_mean", "category_code", "log_amount", "is_round_amount",
    "lat", "long",
]

print("Engineering features...")
train_feat, card_stats = engineer_features(train_df)
score_feat = engineer_features(score_df, card_stats=card_stats)

train_feat["transaction_id"] = train_feat["transaction_id"].astype(str)
score_feat["transaction_id"] = score_feat["transaction_id"].astype(str)
train_feat["is_fraud"] = train_feat["is_fraud"].astype("int64")

train_fg_df = train_feat[FEATURE_COLS + ["is_fraud"]].copy()
print(f"Feature df: {train_fg_df.shape}")

# ── Feature Group ──────────────────────────────────────────────────────────────
print("Setting up feature group cctxn89f322...")

need_insert = False
fg = fs.get_feature_group("cctxn89f322", version=1)
if fg is None:
    try:
        fg = fs.create_feature_group(
            name="cctxn89f322",
            version=1,
            primary_key=["transaction_id"],
            description="Credit card fraud transaction features",
            online_enabled=True,
        )
        need_insert = True
        print("Feature group created.")
    except Exception as e:
        print(f"Create failed ({e}), trying list...")
        fgs = fs.get_feature_groups("cctxn89f322")
        if fgs:
            fg = sorted(fgs, key=lambda x: x.version)[0]
            print(f"Found FG via list: version {fg.version}")
        else:
            raise RuntimeError("Cannot get or create feature group") from e
else:
    print(f"Feature group found (version {fg.version}).")

if need_insert:
    print("Inserting features...")
    fg.insert(train_fg_df, wait=True)
    print("Insert complete.")
else:
    print("Skipping insert (data already present).")

# ── Feature View ───────────────────────────────────────────────────────────────
print("Setting up feature view cctd89f322...")
fv = fs.get_feature_view("cctd89f322", version=1)
if fv is None:
    try:
        fv = fs.create_feature_view(
            name="cctd89f322",
            version=1,
            query=fg.select_all(),
            labels=["is_fraud"],
        )
        print("Feature view created.")
    except Exception as e:
        print(f"FV create failed ({e}), trying list...")
        fvs = fs.get_feature_views("cctd89f322")
        if fvs:
            fv = sorted(fvs, key=lambda x: x.version)[0]
            print(f"Found FV via list: version {fv.version}")
        else:
            raise RuntimeError("Cannot get or create feature view") from e
else:
    print(f"Feature view found (version {fv.version}).")

# ── Training Dataset ───────────────────────────────────────────────────────────
print("Creating training dataset...")
td_version, td_job = fv.create_train_test_split(
    test_size=0.2,
    write_options={"wait_for_job": True},
)
print(f"Training dataset version: {td_version}")

X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=td_version)

# Convert labels to 1D numpy arrays
if hasattr(y_train, "values"):
    y_train = y_train.values.ravel()
else:
    y_train = np.array(y_train).ravel()
if hasattr(y_test, "values"):
    y_test = y_test.values.ravel()
else:
    y_test = np.array(y_test).ravel()

DROP_COLS = ["transaction_id", "cc_num"]
X_train = X_train.drop(columns=[c for c in DROP_COLS if c in X_train.columns])
X_test  = X_test.drop(columns=[c for c in DROP_COLS if c in X_test.columns])
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Fraud rate train: {y_train.mean():.4f}, test: {y_test.mean():.4f}")

# ── Train XGBoost ──────────────────────────────────────────────────────────────
print("Training XGBoost classifier...")
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

neg = int((y_train == 0).sum())
pos = max(int((y_train == 1).sum()), 1)
spw = float(neg) / float(pos)
print(f"Class balance neg={neg} pos={pos} scale_pos_weight={spw:.2f}")

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    scale_pos_weight=spw,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)

y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = float(roc_auc_score(y_test, y_prob))
pr_auc  = float(average_precision_score(y_test, y_prob))
print(f"ROC AUC: {roc_auc:.4f}  PR AUC: {pr_auc:.4f}")

# ── Register model ─────────────────────────────────────────────────────────────
import joblib
model_dir = "/tmp/ccmodel89f322"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, f"{model_dir}/model.pkl")
card_stats.to_csv(f"{model_dir}/card_stats.csv", index=False)

mr = project.get_model_registry()
try:
    existing = mr.get_model("ccmodel89f322", version=1)
    if existing is not None:
        existing.delete()
        print("Deleted existing model version 1.")
except Exception as e:
    print(f"No existing model to delete: {e}")

hw_model = mr.python.create_model(
    name="ccmodel89f322",
    metrics={"roc_auc": roc_auc, "pr_auc": pr_auc},
    description="Credit card fraud XGBoost classifier",
)
hw_model.save(model_dir)
print(f"Model registered: ccmodel89f322 v{hw_model.version}")

# ── Score score_transactions ───────────────────────────────────────────────────
print("Scoring score transactions...")
score_in = score_feat[FEATURE_COLS].copy()
X_score = score_in.drop(columns=[c for c in DROP_COLS if c in score_in.columns])

fraud_prob = model.predict_proba(X_score)[:, 1].astype(float)
pred_df = pd.DataFrame({
    "transaction_id": score_in["transaction_id"].values,
    "fraud_probability": fraud_prob,
})
print(f"Predictions: {pred_df.shape}")
print(pred_df.describe())

# ── Predictions feature group ccpred89f322 ────────────────────────────────────
print("Setting up predictions feature group ccpred89f322...")
pred_fg = fs.get_feature_group("ccpred89f322", version=1)
if pred_fg is None:
    try:
        pred_fg = fs.create_feature_group(
            name="ccpred89f322",
            version=1,
            primary_key=["transaction_id"],
            description="Credit card fraud predictions",
            online_enabled=True,
        )
        print("Predictions FG created.")
    except Exception as e:
        print(f"Pred FG create failed ({e})")
        fgs = fs.get_feature_groups("ccpred89f322")
        if fgs:
            pred_fg = sorted(fgs, key=lambda x: x.version)[0]
        else:
            raise
else:
    print(f"Predictions FG found (version {pred_fg.version}).")

print("Inserting predictions...")
pred_fg.insert(pred_df, wait=True)
print("Predictions inserted.")

print(f"\nPipeline complete! ROC AUC = {roc_auc:.4f}")
