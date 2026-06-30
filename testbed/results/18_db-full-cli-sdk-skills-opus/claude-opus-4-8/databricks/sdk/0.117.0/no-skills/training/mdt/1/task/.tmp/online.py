from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as cat

w = WorkspaceClient()
CAT = 'workspace'
SCH = 'mlpabc1d5e2'
SRC = f'{CAT}.{SCH}.scaledd437a3'
ONLINE = f'{CAT}.{SCH}.scaledd437a3_online'

spec = cat.OnlineTableSpec(
    source_table_full_name=SRC,
    primary_key_columns=['row_id'],
    perform_full_copy=True,
    run_triggered=cat.OnlineTableSpecTriggeredSchedulingPolicy(),
)
ot = cat.OnlineTable(name=ONLINE, spec=spec)
res = w.online_tables.create_and_wait(table=ot, timeout=__import__('datetime').timedelta(minutes=20))
print('online table:', res.name)
print('state:', res.status.detailed_state if res.status else None)
