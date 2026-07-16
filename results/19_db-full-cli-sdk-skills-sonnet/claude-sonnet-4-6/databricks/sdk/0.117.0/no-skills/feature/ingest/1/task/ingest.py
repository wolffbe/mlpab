import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema_fqn = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_fqn.split('.')
wh_id = '4dfab06c923fe3cc'
table_name = 'transactions9dd1da'
full_table = f'{catalog_name}.{schema_name}.{table_name}'
vol_path = f'/Volumes/{catalog_name}/{schema_name}/ingest_data'
staging_table = f'{catalog_name}.{schema_name}.transactions9dd1da_stg'


def run_sql(sql, desc=""):
    label = desc or sql[:80]
    print(f"Running: {label}")
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=wh_id,
        catalog=catalog_name,
        schema=schema_name,
        wait_timeout='50s'
    )
    # Poll until done if still pending/running
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(5)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state == StatementState.SUCCEEDED:
        print(f"  OK")
        if resp.result and resp.result.data_array:
            print(f"  Result: {resp.result.data_array[:3]}")
    else:
        print(f"  FAILED: {resp.status.error}")
        raise RuntimeError(f"SQL failed: {resp.status.error}")
    return resp


# Create the target Delta table with CDF enabled (needed for online tables)
run_sql(f"""
CREATE OR REPLACE TABLE {full_table} (
  row_id STRING NOT NULL,
  account_id STRING,
  event_time BIGINT,
  amount DOUBLE,
  category STRING
)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
)
""", f"CREATE TABLE {full_table}")

# Create staging table with all strings (CSV import)
run_sql(f"""
CREATE OR REPLACE TABLE {staging_table} (
  row_id STRING,
  account_id STRING,
  event_time STRING,
  amount STRING,
  category STRING
)
""", "CREATE staging table")

# COPY INTO staging from both files
run_sql(f"""
COPY INTO {staging_table}
FROM '{vol_path}/transactions_export_1.csv'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false')
COPY_OPTIONS ('mergeSchema' = 'true')
""", "COPY file 1 into staging")

run_sql(f"""
COPY INTO {staging_table}
FROM '{vol_path}/transactions_export_2.csv'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false')
COPY_OPTIONS ('mergeSchema' = 'true')
""", "COPY file 2 into staging")

# Check staging count
run_sql(f"SELECT COUNT(*) as cnt FROM {staging_table}", "Count staging rows")

# Insert deduplicated data into the target table
# Use row_number to keep one row per row_id (any duplicate has the same data)
run_sql(f"""
INSERT INTO {full_table}
SELECT row_id, account_id, CAST(event_time AS BIGINT), CAST(amount AS DOUBLE), category
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY row_id) AS rn
  FROM {staging_table}
)
WHERE rn = 1
""", "Insert deduplicated data into target")

# Verify final count
run_sql(f"SELECT COUNT(*) as cnt FROM {full_table}", "Count final table rows")

# Drop staging table
run_sql(f"DROP TABLE IF EXISTS {staging_table}", "Drop staging table")

print("Data ingestion complete.")
