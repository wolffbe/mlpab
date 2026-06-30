import os, time
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
synced_name = f'{cat}.{sch}.predictionsa834e5_synced'

# settle check
for i in range(20):
    cur = w.database.get_synced_database_table(name=synced_name)
    d = getattr(cur.data_synchronization_status, 'detailed_state', None)
    print('sync state:', d)
    if d and 'NO_PENDING' in str(d) or (d and 'CONTINUOUS' in str(d)):
        break
    if d and 'TRIGGERED' in str(d):
        time.sleep(15); continue
    break

# offline feature table: confirm PK + rows
wh = '4dfab06c923fe3cc'
tbl = f'{cat}.{sch}.predictionsa834e5'
r = w.statement_execution.execute_statement(warehouse_id=wh,
    statement=f'SELECT count(*) FROM {tbl}', wait_timeout='30s')
print('offline rows:', r.result.data_array)
ti = w.tables.get(full_name=tbl)
pks = []
if ti.table_constraints:
    for c in ti.table_constraints:
        if c.primary_key_constraint:
            pks = c.primary_key_constraint.child_columns
print('feature table PK:', pks)

# job present
jname = f'{prefix}_trainjoba834e5'
js = list(w.jobs.list(name=jname))
print('job:', [(j.job_id, j.settings.name) for j in js])
