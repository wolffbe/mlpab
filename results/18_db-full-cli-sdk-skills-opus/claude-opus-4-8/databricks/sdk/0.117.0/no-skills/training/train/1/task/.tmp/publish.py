import time
import databricks.sdk as dsdk
import databricks.sdk.service.ml as ml

w = dsdk.WorkspaceClient()
STORE = "mlpab2138eb-predstore"
SRC = "workspace.mlpab2138eb.predictionsa834e5"
ONLINE_TABLE = "workspace.mlpab2138eb.predictionsa834e5_online"

spec = ml.PublishSpec(
    online_store=STORE,
    online_table_name=ONLINE_TABLE,
    publish_mode=ml.PublishSpecPublishMode.SNAPSHOT,
)
resp = w.feature_store.publish_table(source_table_name=SRC, publish_spec=spec)
print("publish response:", resp)
