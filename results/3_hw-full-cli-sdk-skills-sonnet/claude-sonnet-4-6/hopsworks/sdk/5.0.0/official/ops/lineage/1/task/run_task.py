import hopsworks
import pandas as pd
import json
import os

# Connect to Hopsworks
print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()
print(f"Connected to project: {project.name}")

# Load raw data
print("Loading raw data...")
df_a = pd.read_csv("data/raw_a.csv")
df_b = pd.read_csv("data/raw_b.csv")
print(f"raw_a: {len(df_a)} rows, raw_b: {len(df_b)} rows")

# Create feature group rawa822586
print("Creating feature group rawa822586...")
fg_a = fs.get_or_create_feature_group(
    name="rawa822586",
    version=1,
    primary_key=["row_id"],
    description="Raw table A",
    online_enabled=False,
)
fg_a.insert(df_a, write_options={"wait_for_job": True})
print("rawa822586 inserted")

# Create feature group rawb822586
print("Creating feature group rawb822586...")
fg_b = fs.get_or_create_feature_group(
    name="rawb822586",
    version=1,
    primary_key=["row_id"],
    description="Raw table B",
    online_enabled=False,
)
fg_b.insert(df_b, write_options={"wait_for_job": True})
print("rawb822586 inserted")

# Compute derived table: inner join on row_id, col_sum = a_val + b_val rounded to 6 decimal places
print("Computing derived table...")
df_merged = pd.merge(df_a, df_b, on="row_id", how="inner")
df_merged["col_sum"] = (df_merged["a_val"] + df_merged["b_val"]).round(6)
df_derived = df_merged[["row_id", "col_sum"]].copy()
print(f"derived822586: {len(df_derived)} rows (inner join)")

# Create feature group derived822586 with online storage enabled
print("Creating feature group derived822586...")
fg_derived = fs.get_or_create_feature_group(
    name="derived822586",
    version=1,
    primary_key=["row_id"],
    description="Derived table with col_sum = a_val + b_val, inner join of rawa822586 and rawb822586",
    online_enabled=True,
)
fg_derived.insert(df_derived, write_options={"wait_for_job": True})
print("derived822586 inserted")

# Register lineage: add parents to derived feature group
print("Registering lineage...")
try:
    fg_derived.add_tag("lineage", {"derived_from": ["rawa822586", "rawb822586"]})
    print("Lineage tag added")
except Exception as e:
    print(f"Tag add failed (may not be supported): {e}")

# Try to register explicit lineage/parents if available
try:
    # Check if there's a parents/lineage API
    if hasattr(fg_derived, 'add_parent'):
        fg_derived.add_parent(fg_a)
        fg_derived.add_parent(fg_b)
        print("Parents added via add_parent")
    elif hasattr(fg_derived, 'parents'):
        print(f"fg_derived.parents: {fg_derived.parents}")
    else:
        print("No add_parent method available")
except Exception as e:
    print(f"Parent registration error: {e}")

# Also write derived CSV locally for fallback
os.makedirs("submission", exist_ok=True)
df_derived.to_csv("submission/derived822586.csv", index=False)
print("Written submission/derived822586.csv")

# Write answers.json
answers = {"derived_from": sorted(["rawa822586", "rawb822586"])}
with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)
print(f"Written submission/answers.json: {answers}")

print("DONE")
