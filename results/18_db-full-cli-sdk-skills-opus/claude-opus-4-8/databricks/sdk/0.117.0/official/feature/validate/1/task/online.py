from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy)
import time
w = WorkspaceClient()
C = 'workspace.mlpab6d0586'
SRC = f"{C}.events88b330"
ONLINE = f"{C}.events88b330_online"

spec = OnlineTableSpec(
    source_table_full_name=SRC,
    primary_key_columns=['row_id'],
    timeseries_key='event_time',
    perform_full_copy=True,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
)
try:
    res = w.online_tables.create_and_wait(table=OnlineTable(name=ONLINE, spec=spec), timeout=__import__('datetime').timedelta(minutes=20))
    print('online table created:', res.name)
    print('state:', res.unity_catalog_provisioning_state)
    print('serving url:', res.table_serving_url)
except Exception as e:
    print('online create error:', repr(e))
