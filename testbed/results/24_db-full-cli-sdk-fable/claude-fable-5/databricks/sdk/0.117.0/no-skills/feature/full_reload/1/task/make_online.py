import os
import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

FULL_SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
SOURCE = f"{FULL_SCHEMA}.customers03eedc"
ONLINE = f"{FULL_SCHEMA}.customers03eedc_online"

w = WorkspaceClient()

spec = OnlineTableSpec(
    source_table_full_name=SOURCE,
    primary_key_columns=["row_id"],
    timeseries_key="updated_at",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
    perform_full_copy=True,
)

waiter = w.online_tables.create(table=OnlineTable(name=ONLINE, spec=spec))
print("create requested, waiting for provisioning...")
result = waiter.result(timeout=datetime.timedelta(minutes=25))
print("state:", result.status.detailed_state if result.status else None)
print("name:", result.name)
