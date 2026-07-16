from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy,
)
import os

w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
SRC = "%s.%s.customers302d18" % (cat, sch)
ONLINE = "%s.%s.customers302d18_online" % (cat, sch)

spec = OnlineTableSpec(
    source_table_full_name=SRC,
    primary_key_columns=["row_id"],
    timeseries_key="updated_at",
    perform_full_copy=True,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
)
ot = OnlineTable(name=ONLINE, spec=spec)
print("Creating online table", ONLINE, "...")
res = w.online_tables.create_and_wait(table=ot, timeout=__import__('datetime').timedelta(minutes=40))
print("Online table state:", res.unity_catalog_provisioning_state)
print("status:", res.status)
print("serving url:", res.table_serving_url)
