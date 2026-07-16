from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy,
)

w = WorkspaceClient()
cat, sch = 'workspace', 'mlpabbaf12e'
src = f'{cat}.{sch}.events88b330'
online_name = f'{cat}.{sch}.events88b330_online'

spec = OnlineTableSpec(
    source_table_full_name=src,
    primary_key_columns=['row_id'],
    timeseries_key='event_time',
    perform_full_copy=True,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
)
ot = OnlineTable(name=online_name, spec=spec)
res = w.online_tables.create_and_wait(table=ot)
print('online table:', res.name)
print('state:', res.unity_catalog_provisioning_state)
print('serving url:', res.table_serving_url)
