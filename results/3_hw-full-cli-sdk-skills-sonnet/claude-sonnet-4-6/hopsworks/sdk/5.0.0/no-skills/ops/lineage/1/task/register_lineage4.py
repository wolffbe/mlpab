import hopsworks
import pandas as pd
import json
import inspect

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Get existing feature groups
fg_a = fs.get_feature_group("rawa822586", version=1)
fg_b = fs.get_feature_group("rawb822586", version=1)

# Check what create_feature_group accepts
print("Checking create_feature_group signature...")
try:
    sig = inspect.signature(fs.create_feature_group)
    print(f"create_feature_group signature: {sig}")
except Exception as e:
    print(f"Error: {e}")

# Check FeatureStore methods
print("\nFeatureStore methods with 'create':")
for m in dir(fs):
    if 'create' in m.lower():
        print(f"  {m}")

# Delete the existing derived FG
try:
    fg_derived = fs.get_feature_group("derived822586", version=1)
    print(f"\nDeleting derived822586 (id={fg_derived.id})...")
    fg_derived.delete()
    print("Deleted.")
except Exception as e:
    print(f"Delete error: {e}")

# Read data for reinsertion
raw_a = pd.read_csv("data/raw_a.csv")
raw_b = pd.read_csv("data/raw_b.csv")
merged = pd.merge(raw_a, raw_b, on="row_id", how="inner")
merged["col_sum"] = (merged["a_val"] + merged["b_val"]).round(6)
derived = merged[["row_id", "col_sum"]].copy()

# Recreate with parents
print("\nRecreating derived822586 with parents...")
try:
    fg_derived_new = fs.create_feature_group(
        name="derived822586",
        version=1,
        primary_key=["row_id"],
        description="Derived feature table with col_sum from rawa822586 and rawb822586",
        online_enabled=True,
        parents=[fg_a, fg_b],
    )
    print(f"fg_derived_new parents: {fg_derived_new._parents}")
    fg_derived_new.insert(derived, write_options={"wait_for_job": True})
    print("derived822586 inserted.")
except Exception as e:
    print(f"create with parents error: {e}")
    # Try without parents first
    try:
        fg_derived_new = fs.create_feature_group(
            name="derived822586",
            version=1,
            primary_key=["row_id"],
            description="Derived feature table with col_sum from rawa822586 and rawb822586",
            online_enabled=True,
        )
        fg_derived_new.insert(derived, write_options={"wait_for_job": True})
        print("Created without parents.")
    except Exception as e2:
        print(f"create without parents error: {e2}")
