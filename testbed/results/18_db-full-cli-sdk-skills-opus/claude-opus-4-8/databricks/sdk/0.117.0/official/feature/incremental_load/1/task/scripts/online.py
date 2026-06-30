import sys
sys.path.insert(0, 'scripts')
from common import *
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy,
)

SRC = f'{CAT}.{SCH}.incrementalb48074'
ONLINE = f'{CAT}.{SCH}.incrementalb48074_online'

spec = OnlineTableSpec(
    source_table_full_name=SRC,
    primary_key_columns=['row_id'],
    timeseries_key='event_time',
    perform_full_copy=True,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
)
ot = OnlineTable(name=ONLINE, spec=spec)
res = w.online_tables.create_and_wait(table=ot)
print('online table:', res.name)
print('state:', res.unity_catalog_provisioning_state)
