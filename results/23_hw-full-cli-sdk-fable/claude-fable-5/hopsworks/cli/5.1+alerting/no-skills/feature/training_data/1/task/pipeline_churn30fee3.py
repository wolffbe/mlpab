"""Platform job: ingest churn feature data, build PIT-correct training dataset.

Creates feature view `churntraining30fee3` v1 and materializes training
dataset version 1 with columns:
account_id, label_time, amount, balance, credit_score, tier,
sessions_7d, health_score, churned
"""
import pandas as pd
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

BASE = "Resources/churn30fee3"
paths = {}
for f in [
    "transactions.csv",
    "transactions_late.csv",
    "profiles.csv",
    "activity.csv",
    "account_health.csv",
    "labels.csv",
]:
    paths[f] = dataset_api.download(f"{BASE}/{f}", overwrite=True)

tx = fs.get_feature_group("transactions30fee3", 1)
prof = fs.get_feature_group("profiles30fee3", 1)
act = fs.get_feature_group("activity30fee3", 1)
health = fs.get_feature_group("health30fee3", 1)
labels = fs.get_feature_group("labels30fee3", 1)

print("Inserting transactions (base + late export)...")
tx.insert(pd.read_csv(paths["transactions.csv"]), wait=True)
tx.insert(pd.read_csv(paths["transactions_late.csv"]), wait=True)
print("Inserting profiles...")
prof.insert(pd.read_csv(paths["profiles.csv"]), wait=True)
print("Inserting activity...")
act.insert(pd.read_csv(paths["activity.csv"]), wait=True)
print("Inserting account_health...")
health.insert(pd.read_csv(paths["account_health.csv"]), wait=True)
print("Inserting labels...")
labels.insert(pd.read_csv(paths["labels.csv"]), wait=True)

print("Building point-in-time query and feature view...")
query = (
    labels.select_all()
    .join(tx.select(["amount", "balance"]), on=["account_id"], join_type="left")
    .join(prof.select(["credit_score", "tier"]), on=["account_id"], join_type="left")
    .join(act.select(["sessions_7d"]), on=["account_id"], join_type="left")
    .join(health.select(["health_score"]), on=["account_id"], join_type="left")
)

fv = fs.create_feature_view(
    name="churntraining30fee3",
    version=1,
    query=query,
    labels=["churned"],
    description="Churn training feature view (point-in-time correct)",
)

print("Materializing training dataset version 1...")
version, job = fv.create_training_data(
    description="churntraining30fee3 v1",
    data_format="parquet",
    statistics_config=False,
    write_options={"wait_for_job": True},
)
print("Training dataset version:", version)
