import datetime, os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

w = WorkspaceClient()
catalog, schema = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
src = f"{catalog}.{schema}.predictions178367"
online_name = f"{catalog}.{schema}.predictions178367_online"

spec = OnlineTableSpec(
    source_table_full_name=src,
    primary_key_columns=["row_id"],
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
    perform_full_copy=True,
)
ot = w.online_tables.create_and_wait(
    table=OnlineTable(name=online_name, spec=spec),
    timeout=datetime.timedelta(minutes=30),
)
print("online table:", ot.name)
print("status:", ot.status)
