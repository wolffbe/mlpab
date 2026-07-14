"""
Load features into Databricks feature table, enable online access via Lakebase Synced Table,
query via online/Lakebase path.
"""
import os, json, csv, time, datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from databricks.sdk.service import database as db_service

w = WorkspaceClient()

SCHEMA_FULL = os.environ['MLPAB_DATABRICKS_SCHEMA']
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']
CATALOG = SCHEMA_FULL.split('.')[0]
SCHEMA = SCHEMA_FULL.split('.')[1]
TABLE = 'profilesaa70e4'           # the feature table (Delta / offline store)
ST_TABLE = 'profilesaa70e4_st'    # Synced Database Table (online store in Lakebase)
FULL_TABLE = f'{CATALOG}.{SCHEMA}.{TABLE}'
FULL_ST_TABLE = f'{CATALOG}.{SCHEMA}.{ST_TABLE}'
WAREHOUSE_ID = '4dfab06c923fe3cc'
INSTANCE_NAME = PREFIX.replace('_', '-') + '-profiles'  # mlpabfbfade-profiles

HOST = 'https://' + os.environ['DATABRICKS_HOST'].lstrip('https://').rstrip('/')
TOKEN = os.environ['DATABRICKS_TOKEN']

# ── Read input ────────────────────────────────────────────────────────────────
rows = []
with open('data/features.csv') as fh:
    for r in csv.DictReader(fh):
        rows.append(r)

lookup_keys = []
with open('data/lookup_keys.txt') as fh:
    for line in fh:
        k = line.strip()
        if k:
            lookup_keys.append(k)

print(f'Features: {len(rows)}, Lookup keys: {len(lookup_keys)}')
print(f'Feature table: {FULL_TABLE}')
print(f'Online table (Synced): {FULL_ST_TABLE}')
print(f'Lakebase instance: {INSTANCE_NAME}')


def run_sql(stmt, timeout='50s'):
    resp = w.statement_execution.execute_statement(
        statement=stmt,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout=timeout,
    )
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.error:
        raise RuntimeError(f'SQL failed: {resp.status.error}')
    return resp


# ── Step 1: Ensure feature table (Delta) exists with data ────────────────────
print('\n[1/4] Ensuring feature table exists...')
run_sql(f"""CREATE TABLE IF NOT EXISTS {FULL_TABLE} (
    account_id STRING NOT NULL,
    f1 DOUBLE,
    f2 DOUBLE,
    f3 DOUBLE,
    f4 DOUBLE
) TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')""")

res = run_sql(f'SELECT COUNT(*) FROM {FULL_TABLE}')
cnt = int(res.result.data_array[0][0])
print(f'  Row count: {cnt}')

if cnt < len(rows):
    # Truncate and reload if needed
    run_sql(f'TRUNCATE TABLE {FULL_TABLE}')
    BATCH = 60
    def mk_vals(chunk):
        parts = []
        for r in chunk:
            parts.append("('%s', %s, %s, %s, %s)" % (r['account_id'], r['f1'], r['f2'], r['f3'], r['f4']))
        return ', '.join(parts)
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        run_sql('INSERT INTO ' + FULL_TABLE + ' VALUES ' + mk_vals(chunk))
        print(f'  Inserted rows {start+1}–{start+len(chunk)}')

print(f'  Feature table ready ({len(rows)} rows).')


# ── Step 2: Ensure Lakebase instance is available ─────────────────────────────
print('\n[2/4] Checking Lakebase instance...')
inst = w.database.get_database_instance(INSTANCE_NAME)
print(f'  State: {inst.state}  DNS: {inst.read_write_dns}')
if str(inst.state) != 'DatabaseInstanceState.AVAILABLE':
    raise RuntimeError(f'Lakebase instance not available: {inst.state}')


# ── Step 3: Create Synced Database Table (online store) ───────────────────────
print(f'\n[3/4] Creating Synced Database Table (online store)...')
try:
    synced = w.database.create_synced_database_table(
        synced_table=db_service.SyncedDatabaseTable(
            name=FULL_ST_TABLE,
            database_instance_name=INSTANCE_NAME,
            logical_database_name='public',
            spec=db_service.SyncedTableSpec(
                source_table_full_name=FULL_TABLE,
                primary_key_columns=['account_id'],
                scheduling_policy=db_service.SyncedTableSchedulingPolicy.TRIGGERED,
            ),
        )
    )
    print(f'  Synced table created: {synced}')
except Exception as exc:
    print(f'  create raised: {exc}')
    # Try to get existing
    synced = w.database.get_synced_database_table(name=FULL_ST_TABLE)
    print(f'  Using existing synced table: {synced.data_synchronization_status}')

# Wait for the synced table to become active
print('  Waiting for sync to complete...')
ONLINE_STATES = {
    db_service.SyncedTableState.SYNCED_TABLE_ONLINE,
    db_service.SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE,
    db_service.SyncedTableState.SYNCED_TABLE_ONLINE_CONTINUOUS_UPDATE,
    db_service.SyncedTableState.SYNCED_TABLE_ONLINE_TRIGGERED_UPDATE,
    db_service.SyncedTableState.SYNCED_TABLE_ONLINE_UPDATING_PIPELINE_RESOURCES,
}
FAILED_STATES = {
    db_service.SyncedTableState.SYNCED_TABLE_OFFLINE_FAILED,
    db_service.SyncedTableState.SYNCED_TABLE_ONLINE_PIPELINE_FAILED,
}
deadline = time.time() + 30 * 60
while True:
    synced = w.database.get_synced_database_table(name=FULL_ST_TABLE)
    status = synced.data_synchronization_status
    detailed = status.detailed_state if status else None
    elapsed = int(time.time() - deadline + 30 * 60)
    print(f'  State: {detailed}  elapsed={elapsed}s')
    if detailed in ONLINE_STATES:
        break
    if detailed in FAILED_STATES:
        print(f'  Full status: {status}')
        raise RuntimeError(f'Synced table failed: {detailed}')
    if time.time() > deadline:
        raise TimeoutError(f'Timed out waiting for synced table. Last state: {detailed}')
    time.sleep(20)

print(f'  Synced table is online and ready.')


# ── Step 4: Query lookup keys via Lakebase (online/low-latency path) ──────────
print(f'\n[4/4] Querying feature vectors via online store (Synced Table)...')

# The Synced Table is backed by Lakebase. When queried via SQL,
# the SQL engine reads from Lakebase (not Delta), satisfying the online-path requirement.
# Build IN clause for all lookup keys
keys_sql = ', '.join(f"'{k}'" for k in lookup_keys)
query = f"""
SELECT account_id, f1, f2, f3, f4
FROM {FULL_ST_TABLE}
WHERE account_id IN ({keys_sql})
"""

print(f'  Executing query on Synced Table: {FULL_ST_TABLE}')
res = run_sql(query)

vectors = {}
if res.result and res.result.data_array:
    for row in res.result.data_array:
        aid = row[0]
        f1, f2, f3, f4 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        vectors[aid] = [f1, f2, f3, f4]
    print(f'  Retrieved {len(vectors)} vectors from Synced Table (Lakebase online store).')
else:
    print('  WARNING: No results from Synced Table query')

# Check for missing keys
missing = [k for k in lookup_keys if k not in vectors]
if missing:
    print(f'  WARNING: Missing keys: {missing}')

# ── Write submission ──────────────────────────────────────────────────────────
os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as fh:
    json.dump({'vectors': vectors}, fh, indent=2)

print('\nWritten: submission/answers.json')
sample = {k: vectors[k] for k in list(vectors)[:2]}
print(f'Sample: {sample}')
print(f'\nTotal vectors: {len(vectors)} / {len(lookup_keys)} requested')
