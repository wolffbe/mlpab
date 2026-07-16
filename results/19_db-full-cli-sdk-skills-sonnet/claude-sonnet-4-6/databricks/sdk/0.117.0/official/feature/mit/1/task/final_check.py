"""Final verification of the feature table setup."""
import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog, schema_name = schema.split('.')
WAREHOUSE_ID = "4dfab06c923fe3cc"


def run_sql(statement, timeout_secs=60):
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


print("=" * 60)
print("FINAL VERIFICATION: featuresb1ea93")
print("=" * 60)

# 1. Table exists and has right columns
print("\n1. Table structure:")
table_info = w.tables.get(full_name=f"{schema}.featuresb1ea93")
print(f"   Name: {table_info.full_name}")
print(f"   Type: {table_info.table_type}")
columns = [(c.name, c.type_text) for c in (table_info.columns or [])]
print(f"   Columns: {columns}")
expected = ['row_id', 'account_id', 'event_time', 'amount_usd', 'is_weekend', 'amount_7d']
actual = [c[0] for c in columns]
assert actual == expected, f"Column mismatch: expected {expected}, got {actual}"
print("   [OK] Columns match exactly")

# 2. Row count
r = run_sql(f"SELECT COUNT(*) FROM {schema}.featuresb1ea93")
count = int(r.result.data_array[0][0])
print(f"\n2. Row count: {count}")
assert count == 700, f"Expected 700 rows, got {count}"
print("   [OK] 700 rows")

# 3. Sample data spot check
r = run_sql(f"""
SELECT row_id, account_id, event_time, amount_usd, is_weekend, amount_7d
FROM {schema}.featuresb1ea93
WHERE row_id = 'R00000'
""")
row = r.result.data_array[0] if r.result and r.result.data_array else None
print(f"\n3. R00000: {row}")

# 4. Verify amount_usd computation for R00000
# GBP amount=50.17, fx=1.27 → amount_usd should be 63.7159
r2 = run_sql(f"""
SELECT t.amount, t.currency, fx.fx_rate, ROUND(t.amount * fx.fx_rate, 4) as expected_usd
FROM {schema}.txn_raw t
JOIN {schema}.fx_raw fx ON t.currency = fx.currency
WHERE t.row_id = 'R00000'
""")
if r2.result and r2.result.data_array:
    ref = r2.result.data_array[0]
    print(f"   Reference: amount={ref[0]}, currency={ref[1]}, fx={ref[2]}, expected_usd={ref[3]}")
    print(f"   Actual amount_usd: {row[3]}")

# 5. Feature Engineering properties
print("\n4. Feature Engineering properties:")
props = table_info.properties or {}
for key in ['feature_store.feature_table', 'feature_store.primary_keys', 'feature_store.timestamp_keys']:
    val = props.get(key, 'NOT SET')
    print(f"   {key}: {val}")

# 6. Synced table (online access)
print("\n5. Online access (Synced Table):")
st_info = w.postgres.get_synced_table(name=f"synced_tables/{catalog}.{schema_name}.featuresb1ea93_online")
print(f"   Name: {catalog}.{schema_name}.featuresb1ea93_online")
print(f"   State: {st_info.status.detailed_state}")
print(f"   UC state: {st_info.status.unity_catalog_provisioning_state}")
print(f"   Project: {st_info.status.project}")
print(f"   Message: {st_info.status.message}")

print("\n" + "=" * 60)
print("SUMMARY:")
print(f"  Offline table: {schema}.featuresb1ea93 (Delta, {count} rows)")
print(f"  Online table:  {schema}.featuresb1ea93_online (Lakebase Synced Table)")
print(f"  Lakebase:      projects/mlpab17de0a-feat/branches/production/databases/featuredb")
print("=" * 60)
