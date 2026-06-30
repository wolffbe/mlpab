import os
import time
import csv
import io
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
wh_id = '4dfab06c923fe3cc'
vol_path = f'/Volumes/workspace/mlpab30eb4e/{prefix}_jobfiles'


def run_sql(sql, description=''):
    stmt = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=wh_id,
        wait_timeout='50s'
    )
    while stmt.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        stmt = w.statement_execution.get_statement(stmt.statement_id)
    if stmt.status.state != StatementState.SUCCEEDED:
        raise Exception(f'{description} failed: {stmt.status.error}')
    print(f'{description}: {stmt.status.state}')
    return stmt


# Step 1: Create the feature table
print('Step 1: Creating feature table...')
create_sql = f"""
CREATE OR REPLACE TABLE {schema}.predictions7b586d (
  row_id STRING NOT NULL,
  score DOUBLE,
  CONSTRAINT predictions7b586d_pk PRIMARY KEY (row_id)
)
"""
run_sql(create_sql, 'Create table')

# Step 2: Load data from the volume CSV
print('Step 2: Loading predictions data...')
# Read predictions from volume
resp = w.files.download(f'{vol_path}/predictions.csv')
content = resp.contents.read().decode()
rows = list(csv.DictReader(io.StringIO(content)))
print(f'  Read {len(rows)} rows from predictions.csv')

# Insert in batches using VALUES clause
batch_size = 50
for i in range(0, len(rows), batch_size):
    batch = rows[i:i+batch_size]
    def make_val(r):
        return f"('{r['row_id']}', {r['score']})"
    values = ', '.join(make_val(r) for r in batch)
    insert_sql = f"INSERT INTO {schema}.predictions7b586d (row_id, score) VALUES {values}"
    run_sql(insert_sql, f'Insert batch {i//batch_size + 1}')

print('Step 2 complete: all predictions loaded')

# Step 3: Verify the table
verify_sql = f"SELECT COUNT(*) as cnt FROM {schema}.predictions7b586d"
result = run_sql(verify_sql, 'Verify row count')
rows_count = result.result.data_array[0][0]
print(f'Table has {rows_count} rows')

print('Feature table created successfully!')
