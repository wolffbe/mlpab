import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
FQN = f'{cat}.{sch}.scaledf9e607'
ONLINE = f'{cat}.{sch}.scaledf9e607_online'

spec = OnlineTableSpec(
    source_table_full_name=FQN,
    primary_key_columns=['row_id'],
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
    perform_full_copy=True,
)
waiter = w.online_tables.create(table=OnlineTable(name=ONLINE, spec=spec))
import datetime
res = waiter.result(timeout=datetime.timedelta(minutes=25))
print('online table state:', res.status.detailed_state if res.status else res)
print(res.name)
