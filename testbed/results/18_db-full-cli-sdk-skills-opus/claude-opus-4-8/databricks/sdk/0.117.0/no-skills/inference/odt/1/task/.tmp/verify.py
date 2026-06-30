from databricks.sdk import WorkspaceClient
import os, time

w = WorkspaceClient()
catalog, schema = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
wh = '4dfab06c923fe3cc'
tbl = f'{catalog}.{schema}.scoredbfc4ef'
online_tbl = f'{catalog}.{schema}.scoredbfc4ef_online'
pipeline_id = 'a9283df4-4905-4058-9f5e-8e980bee9e6a'


def sql(s):
    r = w.statement_execution.execute_statement(statement=s, warehouse_id=wh, wait_timeout='50s')
    if r.status.state.value != 'SUCCEEDED':
        raise RuntimeError(f'{r.status.state}: {getattr(r.status, "error", None)}')
    return r.result.data_array


# wait for sync pipeline to finish
for _ in range(60):
    p = w.pipelines.get(pipeline_id)
    st = p.state.value if p.state else None
    latest = w.pipelines.list_pipeline_events(pipeline_id, max_results=1)
    print('pipeline state:', st)
    if st in ('IDLE',):
        break
    time.sleep(10)

print('OFFLINE feature table:')
print('  schema:', sql(f'DESCRIBE {tbl}')[:6])
print('  count:', sql(f'SELECT count(*), count(distinct request_id) FROM {tbl}'))

print('ONLINE synced table low-latency lookup:')
print('  count:', sql(f'SELECT count(*) FROM {online_tbl}'))
print('  lookup Q00002:', sql(f"SELECT * FROM {online_tbl} WHERE request_id='Q00002'"))
