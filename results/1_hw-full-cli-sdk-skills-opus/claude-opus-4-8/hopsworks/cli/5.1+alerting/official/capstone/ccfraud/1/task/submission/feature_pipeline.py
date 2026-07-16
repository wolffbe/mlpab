"""F stage: engineer fraud features and write the labelled history to FG cctxnfe5424."""
import sys
import hopsworks
from hsfs.feature import Feature

project = hopsworks.login()

# Pull the shared feature-engineering module from HopsFS so train and inference
# use byte-identical logic.
ds = project.get_dataset_api()
ds.download("Resources/ccdata/ccfraud_features.py", local_path=".", overwrite=True)
sys.path.insert(0, ".")
import ccfraud_features as fe

fs = project.get_feature_store()

train_df, score_df = fe.load_inputs(project)
df = fe.engineer(train_df, score_df)

out = df[df["__src"] == "train"].copy()
keep = ["transaction_id", "cc_num", "datetime", "is_fraud"] + fe.FEATURES
out = out[keep]
out["is_fraud"] = out["is_fraud"].astype("int64")
out["cc_num"] = out["cc_num"].astype("int64")
print("Feature rows:", out.shape, "fraud rate:", round(out["is_fraud"].mean(), 4))

feature_schema = [
    Feature("transaction_id", "string", description="Unique transaction id (primary key)"),
    Feature("cc_num", "bigint", description="Card number"),
    Feature("datetime", "timestamp", description="Transaction timestamp (event time)"),
    Feature("is_fraud", "bigint", description="Fraud label (1=fraud)"),
    Feature("amount", "double", description="Transaction amount"),
    Feature("amt_log", "double", description="log1p of amount"),
    Feature("hour", "bigint", description="Hour of day"),
    Feature("day_of_week", "bigint", description="Day of week (0=Mon)"),
    Feature("is_night", "bigint", description="1 if before 6am"),
    Feature("cat_fraud_rate", "double", description="Historical fraud rate for the category"),
    Feature("dist_from_home", "double", description="km from card's usual location"),
    Feature("dist_from_prev", "double", description="km from previous transaction"),
    Feature("time_since_prev_min", "double", description="minutes since previous transaction"),
    Feature("speed_kmph", "double", description="implied travel speed from previous txn"),
    Feature("amount_to_avg", "double", description="amount relative to card average"),
    Feature("txn_count_1h", "double", description="same-card transactions in trailing hour"),
]

fg = fs.get_or_create_feature_group(
    name="cctxnfe5424",
    version=1,
    description="Engineered credit-card fraud features from labelled transaction history",
    primary_key=["transaction_id"],
    event_time="datetime",
    features=feature_schema,
    online_enabled=False,
    statistics_config=False,
)
fg.insert(out, wait=True)
print("Inserted into cctxnfe5424. FG id:", fg.id)
