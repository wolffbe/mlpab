"""
Create the featuresb1ea93 feature table on Databricks using SQL warehouse.
"""
import io
import os
import time
import base64

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
user = w.current_user.me().user_name

catalog, schema_name = schema.split('.')
volume_files_path = f"/Volumes/{catalog}/{schema_name}/data_upload"

WAREHOUSE_ID = "4dfab06c923fe3cc"

print(f"Schema: {schema}")
print(f"Prefix: {prefix}")
print(f"Warehouse: {WAREHOUSE_ID}")


def run_sql(statement, wait=True, timeout_secs=300):
    """Execute SQL on the warehouse and wait for completion."""
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=statement,
        catalog=catalog,
        schema=schema_name,
        wait_timeout="0s",  # don't wait in the API; we'll poll
    )
    stmt_id = resp.statement_id
    if not wait:
        return stmt_id

    start = time.time()
    while time.time() - start < timeout_secs:
        result = w.statement_execution.get_statement(stmt_id)
        state = result.status.state
        if state in (StatementState.SUCCEEDED, StatementState.FAILED,
                     StatementState.CANCELED, StatementState.CLOSED):
            if state != StatementState.SUCCEEDED:
                err = result.status.error
                raise RuntimeError(f"SQL failed ({state}): {err}\nSQL: {statement[:300]}")
            return result
        time.sleep(3)
    raise TimeoutError(f"SQL timed out after {timeout_secs}s")


# ── 1. Upload CSV files to volume ──────────────────────────────────────────────
print("\n1. Ensuring CSV files are in the volume...")
for fname in ["transactions.csv", "fx_rates.csv"]:
    remote_path = f"{volume_files_path}/{fname}"
    with open(f"data/{fname}", "rb") as f:
        data = f.read()
    w.files.upload(file_path=remote_path, contents=io.BytesIO(data), overwrite=True)
    print(f"   Uploaded {fname}")

# ── 2. Create staging tables from CSV ─────────────────────────────────────────
print("\n2. Creating staging tables...")

run_sql(f"DROP TABLE IF EXISTS {schema}.txn_raw")
run_sql(f"""
CREATE TABLE {schema}.txn_raw AS
SELECT * FROM read_files(
  '{volume_files_path}/transactions.csv',
  format => 'csv',
  header => 'true',
  inferSchema => 'true'
)
""")
print("   txn_raw table created")

run_sql(f"DROP TABLE IF EXISTS {schema}.fx_raw")
run_sql(f"""
CREATE TABLE {schema}.fx_raw AS
SELECT * FROM read_files(
  '{volume_files_path}/fx_rates.csv',
  format => 'csv',
  header => 'true',
  inferSchema => 'true'
)
""")
print("   fx_raw table created")

# ── 3. Compute feature table ────────────────────────────────────────────────────
print("\n3. Computing features and creating featuresb1ea93 table...")

# Drop existing if any
run_sql(f"DROP TABLE IF EXISTS {schema}.featuresb1ea93")

# is_weekend: dayofweek(UTC): 1=Sunday, 7=Saturday
# amount_7d: rolling 7-day sum over event_time in ms (RANGE window)
# 7 days = 7*24*60*60*1000 = 604800000 ms
create_sql = f"""
CREATE TABLE {schema}.featuresb1ea93
USING DELTA
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
)
AS
WITH joined AS (
  SELECT
    t.row_id,
    t.account_id,
    t.event_time,
    t.amount,
    t.currency,
    CAST(t.amount * f.fx_rate AS DOUBLE) AS amount_usd
  FROM {schema}.txn_raw t
  LEFT JOIN {schema}.fx_raw f ON t.currency = f.currency
),
with_weekend AS (
  SELECT
    *,
    CASE
      WHEN dayofweek(from_unixtime(CAST(event_time AS DOUBLE) / 1000)) IN (1, 7) THEN 1
      ELSE 0
    END AS is_weekend
  FROM joined
),
with_window AS (
  SELECT
    *,
    CAST(
      SUM(amount) OVER (
        PARTITION BY account_id
        ORDER BY CAST(event_time AS LONG)
        RANGE BETWEEN 604800000 PRECEDING AND CURRENT ROW
      ) AS DOUBLE
    ) AS amount_7d
  FROM with_weekend
)
SELECT
  row_id,
  account_id,
  CAST(event_time AS LONG) AS event_time,
  amount_usd,
  CAST(is_weekend AS INT) AS is_weekend,
  amount_7d
FROM with_window
ORDER BY event_time
"""

run_sql(create_sql, timeout_secs=120)
print("   featuresb1ea93 table created!")

# ── 4. Verify the table ────────────────────────────────────────────────────────
print("\n4. Verifying table...")
result = run_sql(f"SELECT COUNT(*) AS cnt FROM {schema}.featuresb1ea93")
rows = result.result.data_array if result.result and result.result.data_array else []
print(f"   Row count: {rows}")

result = run_sql(f"SELECT * FROM {schema}.featuresb1ea93 LIMIT 5")
if result.result and result.result.data_array:
    print("   Sample rows:")
    for row in result.result.data_array:
        print(f"     {row}")

# ── 5. Register as Feature Engineering table via REST API ──────────────────────
print("\n5. Registering as Feature Engineering table...")

# Use the Feature Engineering REST API to register the table
import json
try:
    response = w.api_client.do(
        method="POST",
        path="/api/2.0/feature-store/feature-tables",
        body={
            "name": f"{schema}.featuresb1ea93",
            "primary_keys": [{"name": "row_id", "data_type": "string"}],
            "timestamp_keys": [{"name": "event_time", "data_type": "long"}],
            "description": "Transaction features: amount_usd, is_weekend, amount_7d",
        }
    )
    print(f"   Feature table registered: {response}")
except Exception as e:
    print(f"   Feature store registration note: {e}")
    # The table already exists as a Delta table which is the foundation
    # Try the newer Unity Catalog Feature Engineering API
    try:
        response = w.api_client.do(
            method="POST",
            path="/api/2.1/unity-catalog/feature-tables",
            body={
                "name": f"{schema}.featuresb1ea93",
                "primary_keys": ["row_id"],
                "timestamp_keys": ["event_time"],
            }
        )
        print(f"   Feature table registered (UC): {response}")
    except Exception as e2:
        print(f"   UC feature registration note: {e2}")

# ── 6. Create Online Table for low-latency access ─────────────────────────────
print("\n6. Creating online table for low-latency access...")

from databricks.sdk.service.catalog import (
    OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
)

table_name = f"{catalog}.{schema_name}.featuresb1ea93"
online_table_name = f"{catalog}.{schema_name}.featuresb1ea93_online"

try:
    ot = w.online_tables.create(
        name=online_table_name,
        spec=OnlineTableSpec(
            source_table_full_name=table_name,
            primary_key_columns=["row_id"],
            timeseries_key="event_time",
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
        ),
    )
    print(f"   Online table creation initiated: {online_table_name}")

    # Wait for online table to sync
    print("   Waiting for online table to become active...")
    max_wait_ot = 600
    start_ot = time.time()
    while time.time() - start_ot < max_wait_ot:
        try:
            ot_info = w.online_tables.get(name=online_table_name)
            status = ot_info.status
            if status:
                ds = str(status.detailed_state) if hasattr(status, 'detailed_state') else str(status)
                elapsed = int(time.time() - start_ot)
                print(f"   [{elapsed}s] Online table status: {ds}")
                ds_upper = ds.upper()
                if any(x in ds_upper for x in ['ONLINE', 'ACTIVE', 'SYNCED', 'PROVISIONED']):
                    print("   Online table is ready!")
                    break
                if any(x in ds_upper for x in ['OFFLINE_FAILED', 'PIPELINE_FAILED']):
                    print(f"   Online table issue: {ds}")
                    break
        except Exception as e_ot:
            print(f"   Status check error: {e_ot}")
        time.sleep(20)

except Exception as e:
    print(f"   Online table error: {e}")
    if "already exists" in str(e).lower():
        print("   (online table already exists - OK)")

print(f"\nAll done!")
print(f"Feature table: {catalog}.{schema_name}.featuresb1ea93")
print(f"Online table:  {catalog}.{schema_name}.featuresb1ea93_online")
