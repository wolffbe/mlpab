import os

# Sandbox only allows network via the localhost proxy; NO_PROXY would bypass
# it for the 10.x Hopsworks host, so drop the bypass rules.
for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()
print("project:", project.name, "fs:", fs.name)

SUF = "30fee3"

specs = [
    ("transactions" + SUF, ["data/transactions.csv", "data/transactions_late.csv"], "event_time"),
    ("profiles" + SUF, ["data/profiles.csv"], "event_time"),
    ("activity" + SUF, ["data/activity.csv"], "event_time"),
    ("account_health" + SUF, ["data/account_health.csv"], "event_time"),
    ("labels" + SUF, ["data/labels.csv"], "label_time"),
]

fgs = {}
for name, files, evt in specs:
    fg = fs.get_or_create_feature_group(
        name=name,
        version=1,
        primary_key=["account_id"],
        event_time=evt,
        online_enabled=False,
        description=f"{name} features for churn training",
    )
    fgs[name] = fg
    for f in files:
        df = pd.read_csv(f)
        print(f"inserting {f} into {name}: {df.shape}")
        fg.insert(df, write_options={"wait_for_job": True})
    print(f"done {name}")

labels_fg = fgs["labels" + SUF]
trans_fg = fgs["transactions" + SUF]
prof_fg = fgs["profiles" + SUF]
act_fg = fgs["activity" + SUF]
health_fg = fgs["account_health" + SUF]

query = (
    labels_fg.select_all()
    .join(trans_fg.select(["amount", "balance"]), on=["account_id"])
    .join(prof_fg.select(["credit_score", "tier"]), on=["account_id"])
    .join(act_fg.select(["sessions_7d"]), on=["account_id"])
    .join(health_fg.select(["health_score"]), on=["account_id"])
)

fv = fs.get_or_create_feature_view(
    name="churntraining" + SUF,
    version=1,
    query=query,
    labels=["churned"],
    description="Point-in-time correct churn training features",
)
print("feature view:", fv.name, fv.version)

td_version, td_job = fv.create_training_data(
    description="churn training dataset v1",
    data_format="parquet",
    write_options={"wait_for_job": True},
)
print("training dataset version:", td_version)
print("job:", td_job.name if td_job is not None else None)
