import warnings, urllib3
warnings.filterwarnings("ignore")
urllib3.disable_warnings()
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()

tx_fg     = fs.get_feature_group("ct_transactions_c8f821", version=1)
prof_fg   = fs.get_feature_group("ct_profiles_c8f821", version=1)
act_fg    = fs.get_feature_group("ct_activity_c8f821", version=1)
health_fg = fs.get_feature_group("ct_health_c8f821", version=1)
labels_fg = fs.get_feature_group("ct_labels_c8f821", version=1)

# Point-in-time correct query: labels (event_time = label_time) is the left side.
# Each joined FG (event_time = event_time) is joined as-of label_time automatically.
query = (
    labels_fg.select(["account_id", "label_time", "churned"])
    .join(tx_fg.select(["amount", "balance"]), on=["account_id"])
    .join(prof_fg.select(["credit_score", "tier"]), on=["account_id"])
    .join(act_fg.select(["sessions_7d"]), on=["account_id"])
    .join(health_fg.select(["health_score"]), on=["account_id"])
)

# Sanity-read the query result to verify point-in-time correctness and columns.
df = query.read()
print("QUERY COLUMNS:", list(df.columns))
print("QUERY SHAPE:", df.shape)
print(df.sort_values("account_id").head(8).to_string())

fv = fs.get_or_create_feature_view(
    name="churntrainingc8f821", version=1,
    query=query, description="churn training dataset, point-in-time correct")
print("FV created:", fv.name, fv.version)

td_version, job = fv.create_training_data(
    description="version 1 training dataset", data_format="parquet", write_options={"wait_for_job": True})
print("TD VERSION:", td_version, "JOB:", job)
