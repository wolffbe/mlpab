import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import ml

w = WorkspaceClient()
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = "workspace.mlpab6cf45f"
STORE = f"{PREFIX}-store"
SRC = f"{SCHEMA}.profilesf45007"
ONLINE = f"{SCHEMA}.profilesf45007_online"

spec = ml.PublishSpec(
    online_store=STORE,
    online_table_name=ONLINE,
    publish_mode=ml.PublishSpecPublishMode.TRIGGERED,
)
print("publishing", SRC, "->", ONLINE)
resp = w.feature_store.publish_table(source_table_name=SRC, publish_spec=spec)
print("publish response:", resp)
