"""Verify all components have been created correctly."""
import json
import os
import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
CATALOG = 'workspace'
SCHEMA = 'mlpab9db404'
TABLE_NAME = 'incremental3526e9'
FULL_TABLE = f'{CATALOG}.{SCHEMA}.{TABLE_NAME}'
WH_ID = '4dfab06c923fe3cc'
PREFIX = 'mlpab9db404'
JOB_NAME = f'{PREFIX}_incrementaljob3526e9'


def run_sql(statement):
    resp = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=WH_ID,
        wait_timeout='30s',
        catalog=CATALOG,
        schema=SCHEMA,
    )
    stmt_id = resp.statement_id
    while resp.status and resp.status.state and resp.status.state.value in ('PENDING', 'RUNNING'):
        time.sleep(2)
        resp = w.statement_execution.get_statement(stmt_id)
    state = resp.status.state.value if resp.status and resp.status.state else 'UNKNOWN'
    if state != 'SUCCEEDED':
        raise RuntimeError(f'SQL failed: {resp.status.error}')
    return resp


print("=== VERIFICATION ===\n")

# 1. Check Delta table
print("1. Delta table:")
try:
    tbl = w.tables.get(f'{FULL_TABLE}')
    print(f"   Name: {tbl.full_name}")
    print(f"   Type: {tbl.table_type}")
    cols = [(c.name, c.type_name) for c in (tbl.columns or [])]
    print(f"   Columns: {cols}")
except Exception as e:
    print(f"   ERROR: {e}")

# 2. Check row count
print("\n2. Row count:")
try:
    resp = run_sql(f'SELECT COUNT(*) as cnt FROM {FULL_TABLE}')
    rows = resp.result.data_array if resp.result else []
    print(f"   Rows: {rows[0][0] if rows else 'unknown'}")
except Exception as e:
    print(f"   ERROR: {e}")

# 3. Check online store
print("\n3. Online store:")
try:
    stores = list(w.feature_store.list_online_stores())
    for s in stores:
        print(f"   Store: {s.name}, state: {s.state}")
except Exception as e:
    print(f"   ERROR: {e}")

# 4. Check online table
print("\n4. Online table (published):")
try:
    online_tbl = f'{CATALOG}.{SCHEMA}.{TABLE_NAME}_serving'
    tbl2 = w.tables.get(online_tbl)
    print(f"   Name: {tbl2.full_name}")
    print(f"   Type: {tbl2.table_type}")
except Exception as e:
    print(f"   Note: {e}")

# 5. Check job
print("\n5. Recurring job:")
try:
    jobs = list(w.jobs.list(name=JOB_NAME))
    for j in jobs:
        job_details = w.jobs.get(j.job_id)
        sched = job_details.settings.schedule if job_details.settings else None
        print(f"   Job: {j.job_id} - {j.settings.name}")
        if sched:
            print(f"   Schedule: {sched.quartz_cron_expression} ({sched.timezone_id})")
            print(f"   Pause status: {sched.pause_status}")
except Exception as e:
    print(f"   ERROR: {e}")

# 6. Check answers.json
print("\n6. answers.json:")
if os.path.exists('submission/answers.json'):
    with open('submission/answers.json') as f:
        data = json.load(f)
    print(f"   Content: {data}")
else:
    print("   MISSING!")

print("\n=== DONE ===")
