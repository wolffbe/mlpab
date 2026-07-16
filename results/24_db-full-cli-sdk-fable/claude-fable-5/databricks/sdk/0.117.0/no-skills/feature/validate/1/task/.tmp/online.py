import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

w = WorkspaceClient()
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
SRC = f"{SCHEMA}.events385469"
ONLINE = f"{SCHEMA}.events385469_online"

spec = OnlineTableSpec(
    source_table_full_name=SRC,
    primary_key_columns=["row_id"],
    timeseries_key="event_time",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
    perform_full_copy=True,
)
ot = w.online_tables.create(table=OnlineTable(name=ONLINE, spec=spec))
print("created online table request:", ONLINE)
res = ot.result(timeout=__import__("datetime").timedelta(minutes=25))
print("state:", res.status.detailed_state if res.status else None)
