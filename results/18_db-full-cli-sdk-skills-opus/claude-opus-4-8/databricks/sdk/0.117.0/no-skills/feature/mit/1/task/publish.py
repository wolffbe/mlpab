import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import ml

w = WorkspaceClient()
STORE = "mlpab4bb10d-fs74f1ef"
SRC = "workspace.mlpab4bb10d.features74f1ef"
ONLINE_TABLE = "workspace.mlpab4bb10d.features74f1ef_online"

spec = ml.PublishSpec(
    online_store=STORE,
    online_table_name=ONLINE_TABLE,
    publish_mode=ml.PublishSpecPublishMode.TRIGGERED,
)
res = w.feature_store.publish_table(source_table_name=SRC, publish_spec=spec)
print("publish response:", res.as_dict())
