import json
import pandas as pd
import hopsworks
from hsfs.feature import Feature

# ---- Load inputs (CSV ingestion + lookup list) ----
df = pd.read_csv("data/features.csv")
df["account_id"] = df["account_id"].astype(str)
for c in ["f1", "f2", "f3", "f4"]:
    df[c] = df[c].astype(float)
print("loaded df:", df.shape, list(df.columns))

with open("data/lookup_keys.txt") as fh:
    keys = [line.strip() for line in fh if line.strip()]
print("lookup keys:", len(keys))

# ---- Connect ----
project = hopsworks.login()
fs = project.get_feature_store()

# ---- Create / get online feature group ----
fg = fs.get_or_create_feature_group(
    name="profiles27ba29",
    version=1,
    description="Account feature profiles (f1-f4) for online low-latency serving",
    primary_key=["account_id"],
    features=[
        Feature("account_id", "string", description="Account identifier"),
        Feature("f1", "double", description="Feature 1"),
        Feature("f2", "double", description="Feature 2"),
        Feature("f3", "double", description="Feature 3"),
        Feature("f4", "double", description="Feature 4"),
    ],
    online_enabled=True,
    stream=True,
    statistics_config=False,
)

# ---- Insert (wait for online + offline ingestion) ----
fg.insert(df, wait=True)
print("inserted; fg.id =", fg.id)

# ---- Feature view over the FG for online vector retrieval ----
query = fg.select(["account_id", "f1", "f2", "f3", "f4"])
fv = fs.get_or_create_feature_view(
    name="profiles27ba29_fv",
    version=1,
    description="Online read view for profiles27ba29",
    query=query,
)

fv.init_serving()

# ---- Retrieve each key through the ONLINE read path ----
vectors = {}
for k in keys:
    row = fv.get_feature_vector(entry={"account_id": k}, return_type="pandas")
    # row is a single-row DataFrame with named columns
    vec = [float(row["f1"].iloc[0]), float(row["f2"].iloc[0]),
           float(row["f3"].iloc[0]), float(row["f4"].iloc[0])]
    vectors[k] = vec

print("retrieved", len(vectors), "vectors")
print("sample:", keys[0], vectors[keys[0]])

with open("submission/answers.json", "w") as fh:
    json.dump({"vectors": vectors}, fh, indent=2)
print("wrote submission/answers.json")
