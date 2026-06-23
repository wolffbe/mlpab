import hopsworks
import pandas as pd

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# ─── Load data ────────────────────────────────────────────────────────────────
data_dir = "./data"

transactions = pd.read_csv(f"{data_dir}/transactions.csv")
transactions_late = pd.read_csv(f"{data_dir}/transactions_late.csv")
all_transactions = pd.concat([transactions, transactions_late]).drop_duplicates()

profiles = pd.read_csv(f"{data_dir}/profiles.csv")
activity = pd.read_csv(f"{data_dir}/activity.csv")
account_health = pd.read_csv(f"{data_dir}/account_health.csv")
labels = pd.read_csv(f"{data_dir}/labels.csv")

print(f"Labels: {len(labels)} rows")
print(f"Transactions: {len(all_transactions)} rows")

# Add event_time = label_time for the labels FG
labels_fg_data = labels.copy()
labels_fg_data['event_time'] = labels_fg_data['label_time']
# labels_fg_data has: account_id, label_time, churned, event_time

print("\nSample data shapes:")
print(f"  transactions: {all_transactions.shape}")
print(f"  profiles: {profiles.shape}")
print(f"  activity: {activity.shape}")
print(f"  account_health: {account_health.shape}")
print(f"  labels_fg_data: {labels_fg_data.shape}")

# ─── Create or get feature groups ─────────────────────────────────────────────
print("\n=== Creating Feature Groups ===")

def get_or_create_fg(fs, name, version, primary_key, event_time, df):
    fg = fs.get_feature_group(name, version=version)
    if fg is not None:
        print(f"  {name} v{version}: already exists")
        return fg
    print(f"  {name} v{version}: creating...")
    fg = fs.create_feature_group(
        name=name,
        version=version,
        primary_key=primary_key,
        event_time=event_time,
        description=f"Feature group {name}",
        online_enabled=False,
    )
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"  {name} v{version}: inserted {len(df)} rows")
    return fg


labels_fg = get_or_create_fg(
    fs, "labels_churn", 1,
    primary_key=["account_id"],
    event_time="event_time",
    df=labels_fg_data[["account_id", "event_time", "label_time", "churned"]]
)

transactions_fg = get_or_create_fg(
    fs, "transactions_churn", 1,
    primary_key=["account_id"],
    event_time="event_time",
    df=all_transactions
)

profiles_fg = get_or_create_fg(
    fs, "profiles_churn", 1,
    primary_key=["account_id"],
    event_time="event_time",
    df=profiles
)

activity_fg = get_or_create_fg(
    fs, "activity_churn", 1,
    primary_key=["account_id"],
    event_time="event_time",
    df=activity
)

health_fg = get_or_create_fg(
    fs, "account_health_churn", 1,
    primary_key=["account_id"],
    event_time="event_time",
    df=account_health
)

print("\n=== All feature groups ready ===")

# ─── Build query ───────────────────────────────────────────────────────────────
# Select desired columns: account_id, label_time, churned from labels FG
# Then join features from other FGs (PIT correct since FGs have event_time)
query = (
    labels_fg.select(["account_id", "label_time", "churned"])
    .join(transactions_fg.select(["amount", "balance"]), on=["account_id"])
    .join(profiles_fg.select(["credit_score", "tier"]), on=["account_id"])
    .join(activity_fg.select(["sessions_7d"]), on=["account_id"])
    .join(health_fg.select(["health_score"]), on=["account_id"])
)

print("\nQuery built.")
print("Query string preview:")
try:
    print(query.to_string())
except Exception as e:
    print(f"  (could not print query string: {e})")

# ─── Create feature view ───────────────────────────────────────────────────────
print("\n=== Creating Feature View ===")

FV_NAME = "churntraining2d2b0c"
FV_VERSION = 1

fv = fs.get_feature_view(FV_NAME, version=FV_VERSION)
if fv is not None:
    print(f"Feature view {FV_NAME} v{FV_VERSION} already exists")
else:
    fv = fs.create_feature_view(
        name=FV_NAME,
        version=FV_VERSION,
        query=query,
        labels=["churned"],
        description="Churn prediction training dataset",
    )
    print(f"Feature view {FV_NAME} v{FV_VERSION} created")

# ─── Create training data (version 1) ─────────────────────────────────────────
print("\n=== Creating Training Data ===")

# Check if training data version 1 already exists
existing_tds = fv.get_training_datasets()
existing_versions = [td.version for td in existing_tds] if existing_tds else []
print(f"Existing training dataset versions: {existing_versions}")

if 1 not in existing_versions:
    print("Creating training data version 1...")
    td_version, job = fv.create_training_data(
        description="PIT-correct churn training dataset v1",
        data_format="csv",
        coalesce=True,
        write_options={"wait_for_job": True},
    )
    print(f"Training data created: version={td_version}, job={job}")
else:
    print("Training data version 1 already exists")

# ─── Verify ────────────────────────────────────────────────────────────────────
print("\n=== Verifying ===")
try:
    X, y = fv.get_training_data(training_dataset_version=1)
    print(f"Read back training data: X shape={X.shape}, y={'None' if y is None else y.shape}")
    print(f"X columns: {list(X.columns)}")
    if y is not None:
        print(f"y columns: {list(y.columns)}")
    print(f"\nFirst few rows of X:")
    print(X.head(3))
except Exception as e:
    print(f"Verification error: {e}")

print("\nDone!")
