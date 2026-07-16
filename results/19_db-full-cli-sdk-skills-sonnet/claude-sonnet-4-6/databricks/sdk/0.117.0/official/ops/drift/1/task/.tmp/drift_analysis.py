import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
warehouse_id = '4dfab06c923fe3cc'
catalog, db = schema.split('.')

def run_sql(sql, timeout=300):
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout='50s',
    )
    stmt_id = resp.statement_id
    start = time.time()
    while resp.status.state in [StatementState.PENDING, StatementState.RUNNING]:
        if time.time() - start > timeout:
            raise TimeoutError(f"SQL timeout after {timeout}s")
        time.sleep(2)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {resp.status.error}")
    return resp

def get_rows(resp):
    if not resp.result or not resp.result.data_array:
        return []
    return resp.result.data_array

# Step 1: Create volume
print("Creating volume...")
run_sql(f"CREATE VOLUME IF NOT EXISTS {schema}.{prefix}_vol")
print("Volume created")

# Step 2: Upload CSV to volume
vol_path = f"/Volumes/{catalog}/{db}/{prefix}_vol/features.csv"
local_path = "/Users/wolffbe/workspace/banter/testbed/results/19_db-full-cli-sdk-skills-sonnet/claude-sonnet-4-6/databricks/sdk/0.117.0/official/ops/drift/1/task/data/features.csv"

print("Uploading CSV...")
with open(local_path, 'rb') as f:
    w.files.upload(vol_path, f, overwrite=True)
print("CSV uploaded")

# Step 3: Create table from CSV
table_name = f"{schema}.{prefix}_features"
print(f"Creating table {table_name}...")
run_sql(f"DROP TABLE IF EXISTS {table_name}")
# Create managed table then COPY INTO from volume
run_sql(f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    entity_id STRING,
    event_time TIMESTAMP,
    f1 DOUBLE,
    f2 DOUBLE,
    f3 DOUBLE,
    f4 DOUBLE,
    f5 DOUBLE,
    f6 DOUBLE
)
""")
run_sql(f"""
COPY INTO {table_name}
FROM '{vol_path}'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
COPY_OPTIONS ('mergeSchema' = 'true')
""")
print("Table created")

# Step 4: Verify table
resp = run_sql(f"SELECT COUNT(*) as cnt, MIN(event_time) as min_dt, MAX(event_time) as max_dt FROM {table_name}")
rows = get_rows(resp)
print(f"Table stats: {rows}")

# Step 5: Compute daily statistics per feature
# We'll compute mean and stddev per day for each feature
print("Computing daily statistics...")
resp = run_sql(f"""
SELECT
    DATE(event_time) as dt,
    AVG(f1) as f1_mean, STDDEV(f1) as f1_std,
    AVG(f2) as f2_mean, STDDEV(f2) as f2_std,
    AVG(f3) as f3_mean, STDDEV(f3) as f3_std,
    AVG(f4) as f4_mean, STDDEV(f4) as f4_std,
    AVG(f5) as f5_mean, STDDEV(f5) as f5_std,
    AVG(f6) as f6_mean, STDDEV(f6) as f6_std,
    COUNT(*) as n
FROM {table_name}
GROUP BY DATE(event_time)
ORDER BY dt
""")
rows = get_rows(resp)
print(f"Got {len(rows)} days of statistics")
for row in rows:
    print(','.join(str(x) for x in row))
