import hopsworks
import pandas as pd
import os

# Connect to Hopsworks
print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Load data
data_dir = "./data"
transactions = pd.read_csv(f"{data_dir}/transactions.csv")
transactions_late = pd.read_csv(f"{data_dir}/transactions_late.csv")
profiles = pd.read_csv(f"{data_dir}/profiles.csv")
activity = pd.read_csv(f"{data_dir}/activity.csv")
account_health = pd.read_csv(f"{data_dir}/account_health.csv")
labels = pd.read_csv(f"{data_dir}/labels.csv")

print(f"Labels shape: {labels.shape}")
print(f"Transactions shape: {transactions.shape}")
print(f"Transactions late shape: {transactions_late.shape}")

# Combine transactions
all_transactions = pd.concat([transactions, transactions_late], ignore_index=True)
all_transactions = all_transactions.drop_duplicates()
print(f"Combined transactions shape: {all_transactions.shape}")

# Create feature groups
print("\nCreating feature groups...")

# Transactions feature group
try:
    transactions_fg = fs.get_feature_group("transactions_churn", version=1)
    print("transactions_churn fg already exists")
except:
    transactions_fg = fs.create_feature_group(
        name="transactions_churn",
        version=1,
        primary_key=["account_id"],
        event_time="event_time",
        description="Transaction features for churn prediction",
    )
    transactions_fg.insert(all_transactions, write_options={"wait_for_job": True})
    print("Created and inserted transactions_churn")

# Profiles feature group
try:
    profiles_fg = fs.get_feature_group("profiles_churn", version=1)
    print("profiles_churn fg already exists")
except:
    profiles_fg = fs.create_feature_group(
        name="profiles_churn",
        version=1,
        primary_key=["account_id"],
        event_time="event_time",
        description="Profile features for churn prediction",
    )
    profiles_fg.insert(profiles, write_options={"wait_for_job": True})
    print("Created and inserted profiles_churn")

# Activity feature group
try:
    activity_fg = fs.get_feature_group("activity_churn", version=1)
    print("activity_churn fg already exists")
except:
    activity_fg = fs.create_feature_group(
        name="activity_churn",
        version=1,
        primary_key=["account_id"],
        event_time="event_time",
        description="Activity features for churn prediction",
    )
    activity_fg.insert(activity, write_options={"wait_for_job": True})
    print("Created and inserted activity_churn")

# Account health feature group
try:
    account_health_fg = fs.get_feature_group("account_health_churn", version=1)
    print("account_health_churn fg already exists")
except:
    account_health_fg = fs.create_feature_group(
        name="account_health_churn",
        version=1,
        primary_key=["account_id"],
        event_time="event_time",
        description="Account health features for churn prediction",
    )
    account_health_fg.insert(account_health, write_options={"wait_for_job": True})
    print("Created and inserted account_health_churn")

print("\nAll feature groups ready. Creating feature view...")

# Create feature view with point-in-time correct joins
# Get feature references
transactions_feats = transactions_fg.select(["account_id", "amount", "balance"])
profiles_feats = profiles_fg.select(["credit_score", "tier"])
activity_feats = activity_fg.select(["sessions_7d"])
health_feats = account_health_fg.select(["health_score"])

# Build the query
query = transactions_feats.join(
    profiles_feats, on=["account_id"]
).join(
    activity_feats, on=["account_id"]
).join(
    health_feats, on=["account_id"]
)

print("Query built. Creating feature view...")

# Create or get feature view
try:
    fv = fs.get_feature_view("churntraining2d2b0c_fv", version=1)
    print("Feature view already exists")
except:
    fv = fs.create_feature_view(
        name="churntraining2d2b0c_fv",
        version=1,
        query=query,
        labels=["churned"],
        description="Feature view for churn training dataset",
    )
    print("Feature view created")

print("\nCreating training dataset...")

# Create training dataset using labels
td, job = fv.create_training_data(
    description="Churn training dataset with PIT correct joins",
    data_format="csv",
    label=["churned"],
    write_options={"wait_for_job": True},
    training_dataset_version=1,
)
print(f"Training dataset created: {td}")
print("Done!")
