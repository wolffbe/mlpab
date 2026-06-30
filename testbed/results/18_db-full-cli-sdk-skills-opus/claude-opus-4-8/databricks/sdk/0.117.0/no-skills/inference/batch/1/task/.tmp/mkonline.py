from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as cat

w = WorkspaceClient()

name = 'workspace.mlpabc69812.scores3380ed_online'
spec = cat.OnlineTableSpec(
    source_table_full_name='workspace.mlpabc69812.scores3380ed',
    primary_key_columns=['account_id'],
    perform_full_copy=True,
    run_triggered=cat.OnlineTableSpecTriggeredSchedulingPolicy(),
)
ot = cat.OnlineTable(name=name, spec=spec)
res = w.online_tables.create_and_wait(table=ot)
print('online table:', res.name)
print('state:', res.status.detailed_state if res.status else None)
