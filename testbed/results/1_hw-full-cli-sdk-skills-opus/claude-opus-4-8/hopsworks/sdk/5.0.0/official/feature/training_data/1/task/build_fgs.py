import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import hopsworks
from hsfs.feature import Feature

project = hopsworks.login()
fs = project.get_feature_store()

# ---- Load source CSVs (ingestion only; all real work happens platform-side) ----
tx = pd.read_csv("data/transactions.csv")
tx_late = pd.read_csv("data/transactions_late.csv")
profiles = pd.read_csv("data/profiles.csv")
activity = pd.read_csv("data/activity.csv")
health = pd.read_csv("data/account_health.csv")
labels = pd.read_csv("data/labels.csv")

# enforce types (event_time as bigint epoch ms)
for df in (tx, tx_late):
    df["event_time"] = df["event_time"].astype("int64")
    df["amount"] = df["amount"].astype("float64")
    df["balance"] = df["balance"].astype("float64")
profiles["event_time"] = profiles["event_time"].astype("int64")
profiles["credit_score"] = profiles["credit_score"].astype("int64")
activity["event_time"] = activity["event_time"].astype("int64")
activity["sessions_7d"] = activity["sessions_7d"].astype("int64")
health["event_time"] = health["event_time"].astype("int64")
health["health_score"] = health["health_score"].astype("float64")
labels["label_time"] = labels["label_time"].astype("int64")
labels["churned"] = labels["churned"].astype("int64")

print("rows: tx", len(tx), "tx_late", len(tx_late), "profiles", len(profiles),
      "activity", len(activity), "health", len(health), "labels", len(labels))

# ---- Transactions FG ----
tx_fg = fs.get_or_create_feature_group(
    name="churntrainingc8f821_transactions", version=1,
    description="Account transaction amounts and balances over time",
    primary_key=["account_id"], event_time="event_time",
    online_enabled=False,
    features=[
        Feature("account_id", "string", description="Account identifier"),
        Feature("event_time", "bigint", description="Epoch ms when the value became valid"),
        Feature("amount", "double", description="Transaction amount"),
        Feature("balance", "double", description="Account balance after transaction"),
    ],
)
tx_fg.insert(tx, wait=True)
# transactions_late: later export of the same table, same schema -> append to same FG
tx_fg.insert(tx_late, wait=True)

# ---- Profiles FG ----
prof_fg = fs.get_or_create_feature_group(
    name="churntrainingc8f821_profiles", version=1,
    description="Account credit score and tier over time",
    primary_key=["account_id"], event_time="event_time",
    online_enabled=False,
    features=[
        Feature("account_id", "string", description="Account identifier"),
        Feature("event_time", "bigint", description="Epoch ms when the value became valid"),
        Feature("credit_score", "bigint", description="Credit score"),
        Feature("tier", "string", description="Account tier"),
    ],
)
prof_fg.insert(profiles, wait=True)

# ---- Activity FG ----
act_fg = fs.get_or_create_feature_group(
    name="churntrainingc8f821_activity", version=1,
    description="Account session activity over time",
    primary_key=["account_id"], event_time="event_time",
    online_enabled=False,
    features=[
        Feature("account_id", "string", description="Account identifier"),
        Feature("event_time", "bigint", description="Epoch ms when the value became valid"),
        Feature("sessions_7d", "bigint", description="Sessions in trailing 7 days"),
    ],
)
act_fg.insert(activity, wait=True)

# ---- Account health FG ----
health_fg = fs.get_or_create_feature_group(
    name="churntrainingc8f821_account_health", version=1,
    description="Account health score over time",
    primary_key=["account_id"], event_time="event_time",
    online_enabled=False,
    features=[
        Feature("account_id", "string", description="Account identifier"),
        Feature("event_time", "bigint", description="Epoch ms when the value became valid"),
        Feature("health_score", "double", description="Account health score"),
    ],
)
health_fg.insert(health, wait=True)

# ---- Labels FG (root/spine: event_time = label_time) ----
labels_fg = fs.get_or_create_feature_group(
    name="churntrainingc8f821_labels", version=1,
    description="Churn labels per account at a label time (PIT-join root)",
    primary_key=["account_id"], event_time="label_time",
    online_enabled=False,
    features=[
        Feature("account_id", "string", description="Account identifier"),
        Feature("label_time", "bigint", description="Epoch ms label timestamp"),
        Feature("churned", "bigint", description="1 if account churned"),
    ],
)
labels_fg.insert(labels, wait=True)

print("All feature groups created and populated.")
