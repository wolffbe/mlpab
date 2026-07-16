import hopsworks
import pandas as pd

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Load local data
print("Loading data...")
transactions = pd.concat([
    pd.read_csv("data/transactions.csv"),
    pd.read_csv("data/transactions_late.csv")
], ignore_index=True)
# Deduplicate: keep one row per (account_id, event_time) - last occurrence
transactions = transactions.drop_duplicates(subset=["account_id", "event_time"], keep="last")

profiles = pd.read_csv("data/profiles.csv")
activity = pd.read_csv("data/activity.csv")
account_health = pd.read_csv("data/account_health.csv")
labels = pd.read_csv("data/labels.csv")

print(f"transactions: {len(transactions)} rows")
print(f"profiles: {len(profiles)} rows")
print(f"activity: {len(activity)} rows")
print(f"account_health: {len(account_health)} rows")
print(f"labels: {len(labels)} rows")

# --- Create Feature Groups ---
print("\nCreating feature group: transactions_fg...")
transactions_fg = fs.get_or_create_feature_group(
    name="churn_transactions",
    version=1,
    primary_key=["account_id"],
    event_time="event_time",
    online_enabled=False,
    description="Transaction features for churn model"
)
transactions_fg.insert(transactions, write_options={"wait_for_job": True})
print("transactions_fg inserted.")

print("\nCreating feature group: profiles_fg...")
profiles_fg = fs.get_or_create_feature_group(
    name="churn_profiles",
    version=1,
    primary_key=["account_id"],
    event_time="event_time",
    online_enabled=False,
    description="Profile features for churn model"
)
profiles_fg.insert(profiles, write_options={"wait_for_job": True})
print("profiles_fg inserted.")

print("\nCreating feature group: activity_fg...")
activity_fg = fs.get_or_create_feature_group(
    name="churn_activity",
    version=1,
    primary_key=["account_id"],
    event_time="event_time",
    online_enabled=False,
    description="Activity features for churn model"
)
activity_fg.insert(activity, write_options={"wait_for_job": True})
print("activity_fg inserted.")

print("\nCreating feature group: account_health_fg...")
account_health_fg = fs.get_or_create_feature_group(
    name="churn_account_health",
    version=1,
    primary_key=["account_id"],
    event_time="event_time",
    online_enabled=False,
    description="Account health features for churn model"
)
account_health_fg.insert(account_health, write_options={"wait_for_job": True})
print("account_health_fg inserted.")

# --- Create Feature View ---
print("\nCreating feature view...")
query = transactions_fg.select(["amount", "balance"]) \
    .join(profiles_fg.select(["credit_score", "tier"])) \
    .join(activity_fg.select(["sessions_7d"])) \
    .join(account_health_fg.select(["health_score"]))

try:
    fv = fs.get_feature_view(name="churn_feature_view", version=1)
    print("Feature view already exists, using existing.")
except Exception:
    fv = fs.create_feature_view(
        name="churn_feature_view",
        version=1,
        query=query,
        labels=["churned"],
        description="Feature view for churn training dataset"
    )
    print("Feature view created.")

# --- Create Training Dataset ---
print("\nCreating training dataset with point-in-time correct joins...")
print(f"Labels shape: {labels.shape}")
print(labels.head())

# The labels df needs to have event_time for spine-based point-in-time join
# label_time serves as event_time for spine
labels_spine = labels.rename(columns={"label_time": "event_time"})

try:
    td_version, job = fv.create_training_data(
        description="Churn training dataset with PIT joins",
        data_format="csv",
        write_options={"wait_for_job": True},
        spine=labels_spine,
        primary_key=False,
        event_time=False,
        training_dataset_version=1,
        coalesce=True,
    )
    print(f"Training dataset created: version={td_version}")
except Exception as e:
    print(f"Error with spine approach: {e}")
    # Try alternative API
    try:
        td_version, job = fv.create_training_data(
            description="Churn training dataset with PIT joins",
            data_format="csv",
            write_options={"wait_for_job": True},
            spine=labels_spine,
            training_dataset_version=1,
        )
        print(f"Training dataset created: version={td_version}")
    except Exception as e2:
        print(f"Error with alternative: {e2}")
        raise

print("\nDone!")
