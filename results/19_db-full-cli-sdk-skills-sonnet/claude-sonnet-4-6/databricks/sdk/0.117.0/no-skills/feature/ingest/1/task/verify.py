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
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
online_store_name = f'{prefix}-fs-online'


def run_sql(sql, desc=""):
    label = desc or sql[:80]
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=wh_id,
        catalog=catalog_name,
        schema=schema_name,
        wait_timeout='50s'
    )
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(5)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state == StatementState.SUCCEEDED:
        return resp.result.data_array if resp.result else []
    else:
        print(f"SQL failed: {resp.status.error}")
        return None


print("=== Delta Table Verification ===")
result = run_sql(f"SELECT COUNT(*) FROM {full_table}")
print(f"Total rows: {result}")

result = run_sql(f"SELECT COUNT(DISTINCT row_id) FROM {full_table}")
print(f"Distinct row_ids: {result}")

result = run_sql(f"SELECT MIN(event_time), MAX(event_time) FROM {full_table}")
print(f"event_time range: {result}")

result = run_sql(f"SELECT * FROM {full_table} LIMIT 3")
print(f"Sample rows: {result}")

print("\n=== Table Properties ===")
result = run_sql(f"SHOW TBLPROPERTIES {full_table}")
if result:
    for row in result:
        if 'delta.enableChangeDataFeed' in str(row) or 'primary' in str(row).lower():
            print(f"  {row}")

print("\n=== Online Store ===")
try:
    store = w.feature_store.get_online_store(online_store_name)
    print(f"Online store: {store.name}, state={store.state}, capacity={store.capacity}")
except Exception as e:
    print(f"Error getting online store: {e}")

print("\n=== Unity Catalog Table Info ===")
try:
    tbl = w.tables.get(f'{full_table}')
    print(f"Table type: {tbl.table_type}")
    print(f"Data source format: {tbl.data_source_format}")
    if tbl.columns:
        for col in tbl.columns:
            print(f"  Column: {col.name} ({col.type_name}) nullable={col.nullable}")
except Exception as e:
    print(f"Error getting table info: {e}")
