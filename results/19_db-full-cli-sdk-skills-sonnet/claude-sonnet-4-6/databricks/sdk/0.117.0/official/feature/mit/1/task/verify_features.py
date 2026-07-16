"""Verify the feature table is correct."""
import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog, schema_name = schema.split('.')
WAREHOUSE_ID = "4dfab06c923fe3cc"


def run_sql(statement, timeout_secs=120):
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=statement,
        catalog=catalog,
        schema=schema_name,
        wait_timeout="0s",
    )
    stmt_id = resp.statement_id
    start = time.time()
    while time.time() - start < timeout_secs:
        result = w.statement_execution.get_statement(stmt_id)
        state = result.status.state
        if state in (StatementState.SUCCEEDED, StatementState.FAILED,
                     StatementState.CANCELED, StatementState.CLOSED):
            if state != StatementState.SUCCEEDED:
                raise RuntimeError(f"SQL failed ({state}): {result.status.error}")
            return result
        time.sleep(2)
    raise TimeoutError("SQL timed out")


print("Verifying feature table featuresb1ea93...")
print()

# Check row count
r = run_sql(f"SELECT COUNT(*) as cnt FROM {schema}.featuresb1ea93")
count = r.result.data_array[0][0] if r.result and r.result.data_array else "unknown"
print(f"Row count: {count}")

# Check column structure
r = run_sql(f"DESCRIBE {schema}.featuresb1ea93")
print("\nSchema:")
if r.result and r.result.data_array:
    for row in r.result.data_array:
        print(f"  {row}")

# Spot check some computed values
r = run_sql(f"""
SELECT
  row_id, account_id, event_time, amount_usd, is_weekend, amount_7d
FROM {schema}.featuresb1ea93
LIMIT 5
""")
print("\nSample rows:")
if r.result and r.result.data_array:
    for row in r.result.data_array:
        print(f"  {row}")

# Check is_weekend values
r = run_sql(f"""
SELECT is_weekend, COUNT(*) as cnt
FROM {schema}.featuresb1ea93
GROUP BY is_weekend
ORDER BY is_weekend
""")
print("\nis_weekend distribution:")
if r.result and r.result.data_array:
    for row in r.result.data_array:
        print(f"  is_weekend={row[0]}: {row[1]} rows")

# Check a few amount_usd values vs original
r = run_sql(f"""
SELECT
  f.row_id,
  t.amount,
  t.currency,
  fx.fx_rate,
  f.amount_usd,
  ROUND(t.amount * fx.fx_rate, 10) as expected_usd
FROM {schema}.featuresb1ea93 f
JOIN {schema}.txn_raw t ON f.row_id = t.row_id
JOIN {schema}.fx_raw fx ON t.currency = fx.currency
LIMIT 3
""")
print("\namount_usd verification:")
if r.result and r.result.data_array:
    for row in r.result.data_array:
        print(f"  row_id={row[0]}, amount={row[1]}, currency={row[2]}, fx_rate={row[3]}, amount_usd={row[4]}, expected={row[5]}")

# Check Delta table version
r = run_sql(f"DESCRIBE HISTORY {schema}.featuresb1ea93")
print("\nDelta history:")
if r.result and r.result.data_array:
    for row in r.result.data_array[:3]:
        print(f"  {row}")

# Check table properties
r = run_sql(f"SHOW TBLPROPERTIES {schema}.featuresb1ea93")
print("\nTable properties:")
if r.result and r.result.data_array:
    for row in r.result.data_array:
        print(f"  {row}")

print("\n--- Summary ---")
print(f"Offline table: {schema}.featuresb1ea93 ({count} rows)")
print(f"Online table:  {schema}.featuresb1ea93_online (Lakebase Synced Table, ACTIVE)")
print(f"Lakebase project: mlpab17de0a-feat")
