import warnings
warnings.filterwarnings("ignore")
import json
import pandas as pd
import hopsworks
from hsfs.feature import Feature

with open(".tmp_trino_scores.json") as f:
    rows = json.load(f)

df = pd.DataFrame(rows, columns=["account_id", "score"])
df["account_id"] = df["account_id"].astype(str)
df["score"] = df["score"].astype("float64")
print("scores df:", df.shape, "distinct:", df.account_id.nunique())

project = hopsworks.login()
fs = project.get_feature_store()
hist = fs.get_feature_group("account_feature_history", version=1)

scores = fs.get_or_create_feature_group(
    name="scores30c485",
    version=1,
    description="Batch logistic scores as of T=1773410400000, one row per account",
    primary_key=["account_id"],
    online_enabled=True,
    stream=True,
    parents=[hist],
    statistics_config=False,
    features=[
        Feature("account_id", "string", description="Account identifier"),
        Feature("score", "double", description="sigmoid(w.f + b) as-of-T score, 6dp"),
    ],
)
scores.insert(df, wait=True)
print("inserted, fg id:", scores.id)
