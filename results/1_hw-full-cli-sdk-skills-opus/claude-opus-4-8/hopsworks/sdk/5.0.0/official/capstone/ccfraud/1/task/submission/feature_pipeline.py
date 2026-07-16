"""Feature pipeline (runs locally via SDK; writes feature groups on platform).
Model-independent feature engineering only. Produces:
  - cctxnfe5424  : training features + is_fraud label (offline)
  - ccscorefe5424: score features (offline) for batch inference
  - cctdfe5424   : feature view + materialized training dataset
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import hopsworks
from fe_common import engineer, FEATURES

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")

proj = hopsworks.login()
fs = proj.get_feature_store()

train_raw = pd.read_csv(os.path.join(_DATA, "transactions.csv"))
score_raw = pd.read_csv(os.path.join(_DATA, "score_transactions.csv"))
print("loaded", len(train_raw), "train rows,", len(score_raw), "score rows")

global_rate = float(train_raw["is_fraud"].mean())
fraud_rate_map = train_raw.groupby("category")["is_fraud"].mean().to_dict()
print("global fraud rate", round(global_rate, 4))

# --- training features: lookback over training history only
tr = engineer(train_raw, fraud_rate_map, global_rate)
train_fg_df = tr[["transaction_id", "datetime"] + FEATURES + ["is_fraud"]].copy()
train_fg_df["is_fraud"] = train_fg_df["is_fraud"].astype(int)

# --- score features: lookback over (train + score) history, keep score rows
combined = pd.concat([train_raw.drop(columns=["is_fraud"]), score_raw],
                     ignore_index=True)
sc = engineer(combined, fraud_rate_map, global_rate)
score_ids = set(score_raw["transaction_id"])
sc = sc[sc["transaction_id"].isin(score_ids)].copy()
score_fg_df = sc[["transaction_id", "datetime"] + FEATURES].copy()
assert len(score_fg_df) == len(score_raw), (len(score_fg_df), len(score_raw))
print("engineered. train fg:", train_fg_df.shape, "score fg:", score_fg_df.shape)
print(train_fg_df[FEATURES].describe().T[["mean", "min", "max"]])

# --- feature group: training features
txn_fg = fs.get_or_create_feature_group(
    name="cctxnfe5424", version=1,
    description="Engineered fraud features (velocity, geo distance, amount z-score, "
                "category fraud rate) per card transaction, with is_fraud label.",
    primary_key=["transaction_id"], event_time="datetime",
    online_enabled=False,
)
txn_fg.insert(train_fg_df, write_options={"wait_for_job": True})
print("inserted cctxnfe5424")

# --- feature group: score features (no label)
score_fg = fs.get_or_create_feature_group(
    name="ccscorefe5424", version=1,
    description="Engineered fraud features for the unlabelled scoring slice.",
    primary_key=["transaction_id"], event_time="datetime",
    online_enabled=False,
)
score_fg.insert(score_fg_df, write_options={"wait_for_job": True})
print("inserted ccscorefe5424")

# --- feature view + materialized training dataset
try:
    fs.get_feature_view(name="cctdfe5424", version=1).delete()
except Exception:
    pass
query = txn_fg.select(FEATURES + ["is_fraud"])
fv = fs.create_feature_view(
    name="cctdfe5424", version=1,
    description="Fraud training view over cctxnfe5424 (label is_fraud).",
    query=query, labels=["is_fraud"],
)
print("created feature view cctdfe5424")
version, job = fv.create_training_data(
    description="Training dataset for fraud classifier ccmodelfe5424.",
    write_options={"wait_for_job": True},
)
print("created training dataset version", version)
print("DONE feature_pipeline")
