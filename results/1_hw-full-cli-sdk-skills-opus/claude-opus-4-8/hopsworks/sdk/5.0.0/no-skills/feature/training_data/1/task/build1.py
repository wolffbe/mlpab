import warnings, urllib3
warnings.filterwarnings("ignore")
urllib3.disable_warnings()
import pandas as pd
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()

# --- Load source data (read for ingestion only; all joins/transforms happen platform-side) ---
tx = pd.read_csv("data/transactions.csv")
tx_late = pd.read_csv("data/transactions_late.csv")
tx_all = pd.concat([tx, tx_late], ignore_index=True)
print("transactions combined rows:", len(tx_all))

profiles = pd.read_csv("data/profiles.csv")
activity = pd.read_csv("data/activity.csv")
health = pd.read_csv("data/account_health.csv")
labels = pd.read_csv("data/labels.csv")

for name, df in [("tx", tx_all), ("profiles", profiles), ("activity", activity),
                 ("health", health), ("labels", labels)]:
    print(name, list(df.columns), df.shape)

# --- Create feature groups (offline, event-time aware) ---
tx_fg = fs.get_or_create_feature_group(
    name="ct_transactions_c8f821", version=1,
    description="transactions features", primary_key=["account_id"],
    event_time="event_time", online_enabled=False)
tx_fg.insert(tx_all, wait=True)
print("inserted tx")

prof_fg = fs.get_or_create_feature_group(
    name="ct_profiles_c8f821", version=1,
    description="profile features", primary_key=["account_id"],
    event_time="event_time", online_enabled=False)
prof_fg.insert(profiles, wait=True)
print("inserted profiles")

act_fg = fs.get_or_create_feature_group(
    name="ct_activity_c8f821", version=1,
    description="activity features", primary_key=["account_id"],
    event_time="event_time", online_enabled=False)
act_fg.insert(activity, wait=True)
print("inserted activity")

health_fg = fs.get_or_create_feature_group(
    name="ct_health_c8f821", version=1,
    description="health features", primary_key=["account_id"],
    event_time="event_time", online_enabled=False)
health_fg.insert(health, wait=True)
print("inserted health")

labels_fg = fs.get_or_create_feature_group(
    name="ct_labels_c8f821", version=1,
    description="labels", primary_key=["account_id"],
    event_time="label_time", online_enabled=False)
labels_fg.insert(labels, wait=True)
print("inserted labels")
print("ALL INSERTS DONE")
