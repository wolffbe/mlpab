import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()

schema_full = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab3c8ad7
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]  # mlpab3c8ad7

catalog, schema_name = schema_full.split(".", 1)
print(f"Catalog: {catalog}, Schema: {schema_name}")
print(f"Prefix: {prefix}")

me = w.current_user.me()
user_name = me.user_name
print(f"User: {user_name}")

warehouses = list(w.warehouses.list())
warehouse_id = warehouses[0].id
print(f"Using warehouse: {warehouse_id}")


def run_sql(statement, warehouse_id=warehouse_id, cat=catalog, sch=schema_name):
    """Run a SQL statement and wait for completion."""
    resp = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=warehouse_id,
        catalog=cat,
        schema=sch,
        wait_timeout="50s",
    )
    stmt_id = resp.statement_id
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {resp.status.error}\nStatement: {statement[:300]}")
    return resp


# Step 1: Create volume
print("\nStep 1: Creating volume...")
vol_name = f"{prefix}_data_vol"
run_sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema_name}.{vol_name}")
print(f"Volume {vol_name} ready")

# Step 2: Upload CSVs to the volume
print("\nStep 2: Uploading CSV files...")
vol_path = f"/Volumes/{catalog}/{schema_name}/{vol_name}"

with open("data/transactions.csv", "rb") as f:
    w.files.upload(f"{vol_path}/transactions.csv", f, overwrite=True)
print("Uploaded transactions.csv")

with open("data/fx_rates.csv", "rb") as f:
    w.files.upload(f"{vol_path}/fx_rates.csv", f, overwrite=True)
print("Uploaded fx_rates.csv")

# Step 3: Create the feature table
print("\nStep 3: Creating feature table featuresb1ea93...")

run_sql(f"DROP TABLE IF EXISTS {catalog}.{schema_name}.featuresb1ea93")

sql = f"""
CREATE TABLE {catalog}.{schema_name}.featuresb1ea93
USING DELTA
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
)
AS
WITH transactions AS (
  SELECT
    row_id,
    account_id,
    CAST(event_time AS BIGINT) AS event_time,
    CAST(amount AS DOUBLE) AS amount,
    currency
  FROM read_files(
    '{vol_path}/transactions.csv',
    format => 'csv',
    header => true
  )
),
fx AS (
  SELECT
    currency,
    CAST(fx_rate AS DOUBLE) AS fx_rate
  FROM read_files(
    '{vol_path}/fx_rates.csv',
    format => 'csv',
    header => true
  )
),
joined AS (
  SELECT
    t.row_id,
    t.account_id,
    t.event_time,
    t.amount,
    t.amount * f.fx_rate AS amount_usd,
    CASE WHEN dayofweek(to_timestamp(t.event_time / 1000)) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend
  FROM transactions t
  JOIN fx f ON t.currency = f.currency
)
SELECT
  row_id,
  account_id,
  event_time,
  amount_usd,
  is_weekend,
  SUM(amount) OVER (
    PARTITION BY account_id
    ORDER BY event_time
    RANGE BETWEEN CAST(7 * 24 * 3600 * 1000 AS BIGINT) PRECEDING AND CURRENT ROW
  ) AS amount_7d
FROM joined
"""

run_sql(sql)
print("Feature table created!")

# Verify
resp = run_sql(f"SELECT COUNT(*) as cnt FROM {catalog}.{schema_name}.featuresb1ea93")
cnt = resp.result.data_array[0][0] if resp.result and resp.result.data_array else "unknown"
print(f"Row count: {cnt}")

resp = run_sql(f"SELECT * FROM {catalog}.{schema_name}.featuresb1ea93 LIMIT 3")
if resp.result and resp.result.data_array:
    print("Sample rows:", resp.result.data_array)

# Step 4: Create online table
print("\nStep 4: Creating online table for low-latency access...")

from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

ot_full_name = f"{catalog}.{schema_name}.featuresb1ea93_online"

try:
    existing = w.online_tables.get(ot_full_name)
    print(f"Online table already exists: {existing.name}")
except Exception:
    spec = OnlineTableSpec(
        source_table_full_name=f"{catalog}.{schema_name}.featuresb1ea93",
        primary_key_columns=["row_id"],
        timeseries_key="event_time",
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
    )
    ot = w.online_tables.create(name=ot_full_name, spec=spec)
    print(f"Online table creation started: {ot.name}")

    # Wait for it to be ready
    max_wait = 600
    elapsed = 0
    while elapsed < max_wait:
        ot = w.online_tables.get(ot_full_name)
        state = None
        if ot.status:
            state = str(ot.status.detailed_state) if hasattr(ot.status, 'detailed_state') else str(ot.status)
        print(f"  State ({elapsed}s): {state}")
        if state and any(s in state for s in ["ONLINE", "ACTIVE", "PROVISIONED"]):
            print("Online table is ready!")
            break
        if state and any(s in state for s in ["FAILED", "OFFLINE_FAILED"]):
            print(f"Online table failed: {ot.status}")
            break
        time.sleep(20)
        elapsed += 20

print(f"\nDone! Feature table: {catalog}.{schema_name}.featuresb1ea93")
print(f"Online table: {ot_full_name}")
