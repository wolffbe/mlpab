"""
PySpark job to create training dataset 'churntraining2d2b0c' version 1.
Builds the query natively in the Spark/cluster HSFS environment.
"""
import hopsworks

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Get feature groups
print("Getting feature groups...")
labels_fg = fs.get_feature_group("churn_labels", version=1)
tFG = fs.get_feature_group("churn_transactions", version=1)
pFG = fs.get_feature_group("churn_profiles", version=1)
aFG = fs.get_feature_group("churn_activity", version=1)
hFG = fs.get_feature_group("churn_account_health", version=1)

print("Building query...")
# Build query natively - include all columns from labels_fg (account_id, label_time, churned)
query = labels_fg.select_all() \
    .join(tFG.select(["amount", "balance"])) \
    .join(pFG.select(["credit_score", "tier"])) \
    .join(aFG.select(["sessions_7d"])) \
    .join(hFG.select(["health_score"]))

print("Getting/creating training dataset...")
# Try to get existing TD first
try:
    td = fs.get_training_dataset("churntraining2d2b0c", version=1)
    print(f"Found existing TD: {td.name} v{td.version}")
    print(f"TD schema: {td.schema}")
    # Insert/overwrite with the query
    print("Inserting data with query...")
    td.insert(query, overwrite=True)
    print("Data inserted!")
except Exception as e:
    print(f"Error with existing TD: {e}")
    print("Creating new TD...")
    td = fs.create_training_dataset(
        name="churntraining2d2b0c",
        version=1,
        description="Churn training dataset with PIT joins",
        data_format="csv",
        label=["churned"],
        coalesce=True,
    )
    print("Saving with query...")
    td.save(query)
    print("Saved!")

print("Reading back to verify...")
try:
    df = td.read()
    print(f"Shape: {df.count() if hasattr(df, 'count') else len(df)} rows")
    print(f"Columns: {df.columns}")
except Exception as e:
    print(f"Read error: {e}")

print("Done!")
