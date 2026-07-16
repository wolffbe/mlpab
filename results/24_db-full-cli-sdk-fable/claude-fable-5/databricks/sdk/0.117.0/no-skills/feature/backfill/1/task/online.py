import datetime
import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
TABLE = f"{SCHEMA}.accounts06d84b"
ONLINE = f"{SCHEMA}.accounts06d84b_online"

w = WorkspaceClient()

spec = OnlineTableSpec(
    source_table_full_name=TABLE,
    primary_key_columns=["row_id"],
    timeseries_key="updated_at",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
    perform_full_copy=True,
)
op = w.online_tables.create(table=OnlineTable(name=ONLINE, spec=spec))
print("create requested, waiting for online table to be ready...")
ot = op.result(timeout=datetime.timedelta(minutes=25))
print("online table:", ot.name)
print("status:", ot.status)
