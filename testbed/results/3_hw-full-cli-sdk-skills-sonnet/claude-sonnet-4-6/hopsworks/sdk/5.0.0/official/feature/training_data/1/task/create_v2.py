"""
Create training dataset using feature groups with PIT joins.
Uses labels FG as driving entity, so Hopsworks does PIT joins automatically.
"""
import hopsworks
import pandas as pd

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

labels = pd.read_csv("data/labels.csv")
print(f"Labels: {len(labels)} rows")
print(labels.head(3))

# Create labels feature group
# event_time="label_time" so Hopsworks uses label_time as PIT join timestamp
print("\nCreating labels feature group...")
labels_fg = fs.get_or_create_feature_group(
    name="churn_labels",
    version=1,
    primary_key=["account_id"],
    event_time="label_time",
    online_enabled=False,
    description="Churn labels with label_time as event_time"
)
labels_fg.insert(labels, write_options={"wait_for_job": True})
print("Labels FG inserted.")

# Get existing feature groups
print("\nGetting feature groups...")
tFG = fs.get_feature_group("churn_transactions", version=1)
pFG = fs.get_feature_group("churn_profiles", version=1)
aFG = fs.get_feature_group("churn_activity", version=1)
hFG = fs.get_feature_group("churn_account_health", version=1)

print("FG features:", [f.name for f in labels_fg.features])

# Create feature view with labels as driving entity (PIT joins will happen)
print("\nCreating feature view with labels as driving FG...")
query = labels_fg.select(["churned"]) \
    .join(tFG.select(["amount", "balance"])) \
    .join(pFG.select(["credit_score", "tier"])) \
    .join(aFG.select(["sessions_7d"])) \
    .join(hFG.select(["health_score"]))

# Delete existing FV if needed
try:
    existing = fs.get_feature_view("churn_labels_fv", version=1)
    if existing is not None:
        existing.delete()
        print("Deleted existing FV")
except Exception:
    pass

fv = fs.create_feature_view(
    name="churn_labels_fv",
    version=1,
    query=query,
    description="Churn feature view with PIT joins from labels"
)
print(f"Feature view created: {type(fv)}")
print(f"Features: {[f.name for f in fv.features] if fv else 'None'}")

# Create training dataset
print("\nCreating training dataset 'churntraining2d2b0c'...")
td_version, job = fv.create_training_data(
    description="Churn training dataset v1 with PIT joins",
    data_format="csv",
    write_options={"wait_for_job": True},
    coalesce=True,
)
print(f"Training dataset created! Version: {td_version}")

# Read back to verify
print("\nVerifying training dataset...")
df = fv.get_training_data(training_dataset_version=td_version)
if isinstance(df, tuple):
    df = df[0]
print(f"Shape: {df.shape}")
print(df.head())
print(f"Columns: {list(df.columns)}")
