import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

labels_fg = fs.get_feature_group("churn_labels_fg", version=1)
transactions_fg = fs.get_feature_group("churn_transactions_fg", version=1)
profiles_fg = fs.get_feature_group("churn_profiles_fg", version=1)
activity_fg = fs.get_feature_group("churn_activity_fg", version=1)
health_fg = fs.get_feature_group("churn_health_fg", version=1)

query = (
    labels_fg.select(["account_id", "label_time", "churned"])
    .join(transactions_fg.select(["amount", "balance"]), on=["account_id"])
    .join(profiles_fg.select(["credit_score", "tier"]), on=["account_id"])
    .join(activity_fg.select(["sessions_7d"]), on=["account_id"])
    .join(health_fg.select(["health_score"]), on=["account_id"])
)

fv = fs.create_feature_view(
    name="churntraining2d2b0c",
    version=1,
    query=query,
)

print("Feature view created:", fv.name, "v", fv.version)

td_version, _ = fv.create_training_data(
    description="Churn training dataset v1",
    data_format="parquet",
)

print("Training dataset version:", td_version)
