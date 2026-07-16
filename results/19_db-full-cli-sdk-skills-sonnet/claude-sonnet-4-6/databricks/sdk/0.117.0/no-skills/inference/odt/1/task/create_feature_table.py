import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType, OnlineTableSpec, OnlineTableSpecContinuousSchedulingPolicy
from databricks.sdk.service import sql as dbsql

w = WorkspaceClient()

schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab57e20e
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]   # mlpab57e20e
catalog_schema = schema  # catalog.schema format

# Parse catalog and schema name
parts = schema.split(".")
catalog_name = parts[0]
schema_name = parts[1]

print(f"Schema: {schema}, prefix: {prefix}")

# Step 1: Create a volume for uploading CSV files
volume_name = "input_data"
full_volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"

print(f"Creating volume {schema}.{volume_name}...")
try:
    w.volumes.create(
        catalog_name=catalog_name,
        schema_name=schema_name,
        name=volume_name,
        volume_type=VolumeType.MANAGED
    )
    print("Volume created.")
except Exception as e:
    if "already exists" in str(e).lower():
        print("Volume already exists, continuing.")
    else:
        raise

# Step 2: Upload CSV files to the volume
print("Uploading requests.csv...")
with open("data/requests.csv", "rb") as f:
    w.files.upload(f"{full_volume_path}/requests.csv", f, overwrite=True)
print("Uploading profiles.csv...")
with open("data/profiles.csv", "rb") as f:
    w.files.upload(f"{full_volume_path}/profiles.csv", f, overwrite=True)
print("Files uploaded.")

# Step 3: Use SQL statement execution to create the feature table
# Find a warehouse to run SQL against
warehouses = list(w.warehouses.list())
print(f"Available warehouses: {[wh.name for wh in warehouses]}")
warehouse = warehouses[0]
warehouse_id = warehouse.id
print(f"Using warehouse: {warehouse.name} ({warehouse_id})")

def run_sql(sql_text, warehouse_id):
    print(f"Running SQL: {sql_text[:200]}...")
    response = w.statement_execution.execute_statement(
        statement=sql_text,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
        on_wait_timeout=dbsql.ExecuteStatementRequestOnWaitTimeout.CONTINUE
    )
    statement_id = response.statement_id
    # Poll until done
    while True:
        status = w.statement_execution.get_statement(statement_id)
        state = status.status.state
        if state in (dbsql.StatementState.SUCCEEDED,):
            print(f"  SQL succeeded.")
            return status
        elif state in (dbsql.StatementState.FAILED, dbsql.StatementState.CANCELED, dbsql.StatementState.CLOSED):
            print(f"  SQL failed: {status.status.error}")
            raise RuntimeError(f"SQL failed: {status.status.error}")
        else:
            print(f"  State: {state}, waiting...")
            time.sleep(3)

# Drop existing table if present
run_sql(f"DROP TABLE IF EXISTS {schema}.scored50223c", warehouse_id)

# Create the scored feature table directly from volume CSV files using read_files()
run_sql(f"""
CREATE TABLE {schema}.scored50223c
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
AS
SELECT
  r.request_id,
  r.account_id,
  ROUND(SQRT(POWER(CAST(r.request_lat AS DOUBLE) - CAST(p.home_lat AS DOUBLE), 2) + POWER(CAST(r.request_lon AS DOUBLE) - CAST(p.home_lon AS DOUBLE), 2)), 6) AS distance_deg,
  ROUND(CAST(p.base_score AS DOUBLE) - 0.1 * ROUND(SQRT(POWER(CAST(r.request_lat AS DOUBLE) - CAST(p.home_lat AS DOUBLE), 2) + POWER(CAST(r.request_lon AS DOUBLE) - CAST(p.home_lon AS DOUBLE), 2)), 6), 6) AS score
FROM read_files('{full_volume_path}/requests.csv', format => 'csv', header => true) r
JOIN read_files('{full_volume_path}/profiles.csv', format => 'csv', header => true) p ON r.account_id = p.account_id
""", warehouse_id)

print("Feature table scored50223c created.")

# Verify row count
result = run_sql(f"SELECT COUNT(*) as cnt FROM {schema}.scored50223c", warehouse_id)
print(f"Row count result: {result.result}")

# Step 4: Create an online table for low-latency access
table_full_name = f"{schema}.scored50223c"
online_table_name = f"{schema}.scored50223c"

from databricks.sdk.service.catalog import OnlineTable

print(f"Creating online table for {table_full_name}...")
try:
    online_table_spec = OnlineTableSpec(
        source_table_full_name=table_full_name,
        primary_key_columns=["request_id"],
        run_continuously=OnlineTableSpecContinuousSchedulingPolicy()
    )
    ot = w.online_tables.create(OnlineTable(
        name=online_table_name,
        spec=online_table_spec
    ))
    print(f"Online table creation initiated: {ot}")
except Exception as e:
    if "already exists" in str(e).lower():
        print("Online table already exists.")
    else:
        print(f"Online table creation error: {e}")
        raise

print("Done! Feature table scored50223c created with online access enabled.")
