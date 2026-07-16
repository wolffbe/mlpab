import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

tx = fs.get_feature_group("transactions", 1)
pr = fs.get_feature_group("profiles", 1)
ac = fs.get_feature_group("activity", 1)
ah = fs.get_feature_group("account_health", 1)
lb = fs.get_feature_group("labels", 1)

# Spine = labels (event_time = label_time). Point-in-time LEFT joins pull the
# most-recent feature value at-or-before label_time from each source FG.
query = (
    lb.select(["churned"])
    .join(tx.select(["amount", "balance"]), on=["account_id"], join_type="left")
    .join(pr.select(["credit_score", "tier"]), on=["account_id"], join_type="left")
    .join(ac.select(["sessions_7d"]), on=["account_id"], join_type="left")
    .join(ah.select(["health_score"]), on=["account_id"], join_type="left")
)

name = "churntraining4f7ce4"
# Remove any prior version to keep a clean version 1.
try:
    existing = fs.get_feature_view(name=name, version=1)
    existing.delete()
    print("deleted existing feature view", name)
except Exception as e:
    print("no existing fv to delete:", repr(e))

fv = fs.create_feature_view(
    name=name,
    version=1,
    query=query,
    labels=["churned"],
    description="churn training feature view (exact 9-column PIT spine)",
)
print("created feature view:", fv.name, "v", fv.version)
print("schema:", [f.name for f in fv.schema])
