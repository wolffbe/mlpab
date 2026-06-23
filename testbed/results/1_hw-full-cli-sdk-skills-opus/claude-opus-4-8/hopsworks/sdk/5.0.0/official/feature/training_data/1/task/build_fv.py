import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

labels_fg = fs.get_feature_group("churntrainingc8f821_labels", version=1)
tx_fg = fs.get_feature_group("churntrainingc8f821_transactions", version=1)
prof_fg = fs.get_feature_group("churntrainingc8f821_profiles", version=1)
act_fg = fs.get_feature_group("churntrainingc8f821_activity", version=1)
health_fg = fs.get_feature_group("churntrainingc8f821_account_health", version=1)

# labels FG is the PIT-join root: its event_time (label_time) is the reference.
# Joined FGs contribute their MOST RECENT value with event_time <= label_time.
query = (
    labels_fg.select(["account_id", "label_time", "churned"])
    .join(tx_fg.select(["amount", "balance"]), on=["account_id"], join_type="left")
    .join(prof_fg.select(["credit_score", "tier"]), on=["account_id"], join_type="left")
    .join(act_fg.select(["sessions_7d"]), on=["account_id"], join_type="left")
    .join(health_fg.select(["health_score"]), on=["account_id"], join_type="left")
)

fv = fs.get_or_create_feature_view(
    name="churntrainingc8f821", version=1,
    description="Point-in-time-correct churn features joined to labels",
    query=query,
    labels=["churned"],
)
print("Feature view created:", fv.name, "v", fv.version)

# Materialize training dataset version 1 (point-in-time correct join performed here).
version, job = fv.create_training_data(
    description="PIT-correct churn training data v1",
    data_format="parquet",
    write_options={"wait_for_job": True},
)
print("Training dataset version:", version)
