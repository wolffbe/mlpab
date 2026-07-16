import datetime
import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]

w = WorkspaceClient()

online = OnlineTable(
    name=f"{SCHEMA}.scored288ecf_online",
    spec=OnlineTableSpec(
        source_table_full_name=f"{SCHEMA}.scored288ecf",
        primary_key_columns=["request_id"],
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
        perform_full_copy=True,
    ),
)

result = w.online_tables.create_and_wait(table=online, timeout=datetime.timedelta(minutes=30))
print("online table:", result.name)
print("state:", result.status.detailed_state if result.status else None)
print("serving url:", result.table_serving_url)
