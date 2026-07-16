import hopsworks
import pandas as pd
import json
import os

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

print("Reading features CSV...")
df = pd.read_csv("data/features.csv")
print(f"Loaded {len(df)} rows")

# Read lookup keys
with open("data/lookup_keys.txt") as f:
    lookup_keys = [line.strip() for line in f if line.strip()]
print(f"Lookup keys: {len(lookup_keys)}")

fg_name = "profiles501d2f"
fg_version = 1

print(f"Setting up feature group '{fg_name}' v{fg_version}...")
fg = fs.get_or_create_feature_group(
    name=fg_name,
    version=fg_version,
    primary_key=["account_id"],
    online_enabled=True,
    description="Account feature profiles"
)

print("Inserting data...")
fg.insert(df, write_options={"wait_for_job": True})
print("Data inserted.")

# Create feature view for online serving
print("Setting up feature view...")
q = fg.select_all()
fv = fs.get_or_create_feature_view(
    name=f"{fg_name}_fv",
    version=fg_version,
    query=q
)
print("Feature view ready.")

# Retrieve feature vectors via online store
# The FG features are: account_id, f1, f2, f3, f4
# get_feature_vector returns list in that order; strip account_id
print("Retrieving feature vectors from online store...")
vectors = {}
for key in lookup_keys:
    vec = fv.get_feature_vector({"account_id": key})
    # vec = [account_id, f1, f2, f3, f4] — drop account_id
    feature_vals = [v for v in vec if not isinstance(v, str)]
    vectors[key] = feature_vals
    print(f"  {key}: {feature_vals}")

print(f"Retrieved {len(vectors)} vectors")

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"vectors": vectors}, f)
print("Written submission/answers.json")
