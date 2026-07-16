"""
Query the online synced table (Lakebase) for each lookup key and write answers.json.
"""
import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout

w = WorkspaceClient()
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']
DB_CATALOG_NAME = f'{PREFIX}_dbcat'
TABLE_NAME = 'profilesaa70e4'
SYNCED_TABLE_NAME = f'{DB_CATALOG_NAME}.public.{TABLE_NAME}'
WAREHOUSE_ID = '4dfab06c923fe3cc'


def exec_sql(sql):
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout='50s',
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
    )
    stmt_id = resp.statement_id
    while resp.status.state.value in ('PENDING', 'RUNNING'):
        time.sleep(2)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state.value != 'SUCCEEDED':
        raise RuntimeError(f'SQL failed: {resp.status.error}')
    return resp


# Read lookup keys
print("Reading lookup keys...")
with open('data/lookup_keys.txt', 'r') as f:
    lookup_keys = [line.strip() for line in f if line.strip()]
print(f"Found {len(lookup_keys)} lookup keys: {lookup_keys[:5]}...")

# Query online table (Lakebase synced table) for each lookup key
print(f"\nQuerying online table {SYNCED_TABLE_NAME}...")
keys_list = ', '.join([f"'{k}'" for k in lookup_keys])
query_sql = f"""
SELECT account_id, f1, f2, f3, f4
FROM {SYNCED_TABLE_NAME}
WHERE account_id IN ({keys_list})
ORDER BY account_id
"""

resp = exec_sql(query_sql)
data_array = resp.result.data_array if resp.result else []
print(f"Got {len(data_array)} rows from online table")

# Build vectors dict
vectors = {}
for row in data_array:
    acc_id = row[0]
    f1 = float(row[1])
    f2 = float(row[2])
    f3 = float(row[3])
    f4 = float(row[4])
    vectors[acc_id] = [f1, f2, f3, f4]

print(f"Retrieved {len(vectors)} vectors")

# Verify all keys are present
missing = [k for k in lookup_keys if k not in vectors]
if missing:
    print(f"WARNING: Missing keys: {missing}")

# Write output
os.makedirs('submission', exist_ok=True)
output = {'vectors': vectors}
with open('submission/answers.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nWritten submission/answers.json with {len(vectors)} entries")

# Show sample
for k in list(vectors.keys())[:3]:
    print(f"  {k}: {vectors[k]}")
