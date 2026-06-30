from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'
SCH = 'workspace.mlpab0cfb32'


def sql(q):
    r = w.statement_execution.execute_statement(
        warehouse_id=WH, statement=q, wait_timeout='50s')
    st = r.status.state.value
    if st != 'SUCCEEDED':
        raise RuntimeError(f'{st}: {r.status.error}')
    return r.result.data_array if r.result else None


# load
sql(f"""CREATE OR REPLACE TABLE {SCH}.features AS
SELECT * FROM read_files('/Volumes/workspace/mlpab0cfb32/drift_data/features.csv',
  format=>'csv', header=>true, inferSchema=>true)""")
print('loaded')
print(sql(f"SELECT count(*), min(event_time), max(event_time) FROM {SCH}.features"))
