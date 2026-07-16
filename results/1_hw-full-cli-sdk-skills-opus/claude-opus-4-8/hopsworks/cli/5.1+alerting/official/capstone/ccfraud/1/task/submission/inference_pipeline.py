"""I stage: score every row of score_transactions.csv with ccmodelfe5424 and
write fraud probabilities to the online+offline feature table ccpredfe5424."""
import sys
import json
import joblib
import numpy as np
import pandas as pd
import hopsworks
from hsfs.feature import Feature

project = hopsworks.login()

ds = project.get_dataset_api()
ds.download("Resources/ccdata/ccfraud_features.py", local_path=".", overwrite=True)
sys.path.insert(0, ".")
import ccfraud_features as fe

fs = project.get_feature_store()

# Same feature engineering as training (identical inputs => no skew).
train_df, score_df = fe.load_inputs(project)
df = fe.engineer(train_df, score_df)
score = df[df["__src"] == "score"].copy()

# Load the registered model.
mr = project.get_model_registry()
hw_model = mr.get_model("ccmodelfe5424", version=1)
model_path = hw_model.download()
model = joblib.load(f"{model_path}/model.pkl")
features = json.load(open(f"{model_path}/feature_names.json"))

X = score[features]
score["fraud_probability"] = model.predict_proba(X)[:, 1].astype("float64")
score["fraud_probability"] = score["fraud_probability"].clip(0.0, 1.0)

out = pd.DataFrame({
    "transaction_id": score["transaction_id"].astype(str).values,
    "cc_num": score["cc_num"].astype("int64").values,
    "datetime": pd.to_datetime(score["datetime"], utc=True).values,
    "fraud_probability": score["fraud_probability"].values,
})
print("Scoring rows:", out.shape,
      "prob range:", round(float(out.fraud_probability.min()), 4),
      "-", round(float(out.fraud_probability.max()), 4),
      "mean:", round(float(out.fraud_probability.mean()), 4))

pred_fg = fs.get_or_create_feature_group(
    name="ccpredfe5424",
    version=1,
    description="Predicted fraud probabilities for the scoring transactions",
    primary_key=["transaction_id"],
    event_time="datetime",
    features=[
        Feature("transaction_id", "string", description="Transaction id (primary key)"),
        Feature("cc_num", "bigint", description="Card number"),
        Feature("datetime", "timestamp", description="Transaction timestamp"),
        Feature("fraud_probability", "double", description="Predicted P(fraud) in [0,1]"),
    ],
    online_enabled=True,
    stream=True,
    statistics_config=False,
)
pred_fg.insert(out, wait=True)
print("Inserted into ccpredfe5424. FG id:", pred_fg.id)
