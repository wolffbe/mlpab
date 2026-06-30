import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy,
)

w = WorkspaceClient()
CATALOG, SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
SRC = f"{CATALOG}.{SCHEMA}.scoredbfc4ef"
ONLINE = f"{CATALOG}.{SCHEMA}.scoredbfc4ef_online"

spec = OnlineTableSpec(
    source_table_full_name=SRC,
    primary_key_columns=["request_id"],
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
)
print("creating online table", ONLINE)
res = w.online_tables.create_and_wait(table=OnlineTable(name=ONLINE, spec=spec))
print("name:", res.name)
print("uc provisioning state:", res.unity_catalog_provisioning_state)
print("status:", res.status)
print("serving url:", res.table_serving_url)
print("DONE online")
