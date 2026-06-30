import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

w = WorkspaceClient()
SRC = "workspace.mlpab4bb10d.features74f1ef"
ONLINE = "workspace.mlpab4bb10d.features74f1ef_online"

spec = catalog.OnlineTableSpec(
    source_table_full_name=SRC,
    primary_key_columns=["row_id"],
    timeseries_key="event_time",
    run_triggered=catalog.OnlineTableSpecTriggeredSchedulingPolicy(),
    perform_full_copy=True,
)
ot = catalog.OnlineTable(name=ONLINE, spec=spec)
res = w.online_tables.create(table=ot)
print("create submitted:", res.response.name if hasattr(res, "response") else res)

# wait for active
final = w.online_tables.wait_get_online_table_active(name=ONLINE, timeout=__import__("datetime").timedelta(minutes=20))
print("state:", final.unity_catalog_provisioning_state)
print("serving_url:", final.table_serving_url)
print("spec pk:", final.spec.primary_key_columns, "ts:", final.spec.timeseries_key)
