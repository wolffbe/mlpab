import hopsworks
import pandas as pd
import json
import os

# Connect
project = hopsworks.login()
fs = project.get_feature_store()

# Load CSV
df = pd.read_csv("data/features.csv")
print(f"Loaded {len(df)} rows from CSV")
print(df.head())

# Create/get feature group with online enabled
fg = fs.get_or_create_feature_group(
    name="profiles501d2f",
    version=1,
    primary_key=["account_id"],
    online_enabled=True,
    description="Account feature profiles",
)

print("Inserting data into feature group...")
fg.insert(df, wait=True)
print("Insert complete.")

# Create a feature view
try:
    fv = fs.get_feature_view("profiles501d2f_fv", version=1)
    if fv is None:
        raise ValueError("returned None")
    print("Got existing feature view.")
except Exception as e:
    print(f"Feature view not found ({e}), creating...")
    fv = fs.create_feature_view(
        name="profiles501d2f_fv",
        version=1,
        query=fg.select(["account_id", "f1", "f2", "f3", "f4"]),
    )
    print("Created feature view.")

# Init serving (no training dataset needed for pure online retrieval)
try:
    fv.init_serving()
    print("init_serving() done.")
except Exception as e:
    print(f"init_serving() failed: {e}, trying with training_dataset_version=1")
    try:
        fv.init_serving(training_dataset_version=1)
        print("init_serving(1) done.")
    except Exception as e2:
        print(f"init_serving(1) also failed: {e2}, proceeding without init")

# Read lookup keys
with open("data/lookup_keys.txt") as f:
    keys = [line.strip() for line in f if line.strip()]
print(f"Looking up {len(keys)} keys")

# Retrieve via online store
vectors = {}
for account_id in keys:
    vec = fv.get_feature_vector(
        entry={"account_id": account_id},
        return_type="list",
    )
    # vec may include account_id as the first element; drop any string prefix
    clean = [v for v in vec if not isinstance(v, str)]
    vectors[account_id] = clean
    print(f"  {account_id}: {vec}")

# Write output
os.makedirs("submission", exist_ok=True)
result = {"vectors": vectors}
with open("submission/answers.json", "w") as f:
    json.dump(result, f)
print("Written submission/answers.json")
