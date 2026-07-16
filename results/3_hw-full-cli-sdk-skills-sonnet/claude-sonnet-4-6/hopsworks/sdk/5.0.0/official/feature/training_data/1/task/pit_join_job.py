"""
PIT join job - runs on Hopsworks cluster.
Creates training dataset 'churntraining2d2b0c' version 1
with point-in-time correct joins.
"""
import pandas as pd
import hopsworks
import os

project = hopsworks.login()
fs = project.get_feature_store()

# Read input CSVs from HopsFS
ds = project.get_dataset_api()

def read_hopsfs_csv(path):
    local = f"/tmp/{os.path.basename(path)}"
    ds.download(path, local_path=local, overwrite=True)
    return pd.read_csv(local)

transactions = pd.concat([
    read_hopsfs_csv("Resources/churn_data/transactions.csv"),
    read_hopsfs_csv("Resources/churn_data/transactions_late.csv"),
], ignore_index=True).drop_duplicates(subset=["account_id", "event_time"], keep="last")

profiles = read_hopsfs_csv("Resources/churn_data/profiles.csv")
activity = read_hopsfs_csv("Resources/churn_data/activity.csv")
account_health = read_hopsfs_csv("Resources/churn_data/account_health.csv")
labels = read_hopsfs_csv("Resources/churn_data/labels.csv")

print(f"Loaded: transactions={len(transactions)}, profiles={len(profiles)}, "
      f"activity={len(activity)}, health={len(account_health)}, labels={len(labels)}")


def pit_join_latest(spine_df, feature_df, feature_cols, label_time_col="label_time"):
    """
    For each row in spine_df, find the most recent row in feature_df where
    feature_df.event_time <= spine_df.label_time, joined on account_id.
    """
    result = spine_df.copy()
    for col in feature_cols:
        result[col] = None

    # For each account in spine, get PIT-correct feature value
    for idx, row in spine_df.iterrows():
        acct = row["account_id"]
        t = row[label_time_col]
        candidates = feature_df[(feature_df["account_id"] == acct) &
                                 (feature_df["event_time"] <= t)]
        if len(candidates) > 0:
            latest = candidates.sort_values("event_time").iloc[-1]
            for col in feature_cols:
                result.at[idx, col] = latest[col]

    return result


print("Performing PIT joins...")

# Start with labels
result = labels.copy()

# Join transactions (amount, balance)
result = pit_join_latest(result, transactions, ["amount", "balance"])

# Join profiles (credit_score, tier)
result = pit_join_latest(result, profiles, ["credit_score", "tier"])

# Join activity (sessions_7d)
result = pit_join_latest(result, activity, ["sessions_7d"])

# Join account_health (health_score)
result = pit_join_latest(result, account_health, ["health_score"])

# Select required columns in order
result = result[["account_id", "label_time", "amount", "balance",
                 "credit_score", "tier", "sessions_7d", "health_score", "churned"]]

print(f"Result shape: {result.shape}")
print(result.head())
print(f"Null counts:\n{result.isnull().sum()}")

# Save to local CSV first
result.to_csv("/tmp/churntraining2d2b0c.csv", index=False)

# Create training dataset on platform
print("\nCreating training dataset on platform...")
td = fs.create_training_dataset(
    name="churntraining2d2b0c",
    version=1,
    description="Churn training dataset with PIT joins",
    data_format="csv",
    label=["churned"],
    coalesce=True,
)
td.save(result)
print("Training dataset saved!")
print(f"Training dataset: {td}")
