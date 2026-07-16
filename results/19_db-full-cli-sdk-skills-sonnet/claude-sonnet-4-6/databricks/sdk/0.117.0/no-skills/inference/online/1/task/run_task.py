"""
Task: Load feature profiles into a Databricks online table and retrieve features.
Uses Lakebase (Synced Tables) since Online Tables are deprecated.
"""
import os
import json
import time
import csv
import datetime
import requests

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as cat_service
from databricks.sdk.service import database as db_service
from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout

# Configuration
SCHEMA = os.environ['MLPAB_DATABRICKS_SCHEMA']  # workspace.mlpab483bdc
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']   # mlpab483bdc
HOST = os.environ['DATABRICKS_HOST']
TOKEN = os.environ['DATABRICKS_TOKEN']
TABLE_NAME = 'profilesaa70e4'
FULL_TABLE_NAME = f'{SCHEMA}.{TABLE_NAME}'
WAREHOUSE_ID = '4dfab06c923fe3cc'

# Parse catalog and schema from SCHEMA
CATALOG = SCHEMA.split('.')[0]   # workspace
SCHEMA_NAME = SCHEMA.split('.')[1]  # mlpab483bdc

# Lakebase names (must be DNS compliant: alphanumeric + hyphens only)
DB_INSTANCE_NAME = f'{PREFIX}-lakebase'
DB_CATALOG_NAME = f'{PREFIX}_dbcat'
DB_NAME = 'features_db'

w = WorkspaceClient()

# Step 1: Read the CSV data
print("Reading features.csv...")
features = {}
with open('data/features.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        features[row['account_id']] = {
            'f1': float(row['f1']),
            'f2': float(row['f2']),
            'f3': float(row['f3']),
            'f4': float(row['f4']),
        }
print(f"Loaded {len(features)} rows")

# Step 2: Create Delta table using SQL (offline store / feature table)
print(f"\nCreating Delta table {FULL_TABLE_NAME}...")

def exec_sql(sql, wait=True):
    """Execute SQL statement and return results."""
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout='50s',
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
    )
    stmt_id = resp.statement_id
    # Poll until done
    while resp.status.state.value in ('PENDING', 'RUNNING'):
        time.sleep(2)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state.value != 'SUCCEEDED':
        raise RuntimeError(f"SQL failed: {resp.status.error}")
    return resp

# Drop table if exists, then create
exec_sql(f"DROP TABLE IF EXISTS {FULL_TABLE_NAME}")

exec_sql(f"""
CREATE TABLE {FULL_TABLE_NAME} (
    account_id STRING NOT NULL,
    f1 DOUBLE,
    f2 DOUBLE,
    f3 DOUBLE,
    f4 DOUBLE
)
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
print("Table created.")

# Step 3: Insert data via VALUES
print("Inserting data...")
values_parts = []
for acc_id, vals in features.items():
    values_parts.append(f"('{acc_id}', {vals['f1']}, {vals['f2']}, {vals['f3']}, {vals['f4']})")

values_sql = f"INSERT INTO {FULL_TABLE_NAME} VALUES {', '.join(values_parts)}"
exec_sql(values_sql)
print("Data inserted.")

# Step 4: Create Lakebase database instance for online/low-latency access
print(f"\nCreating Lakebase database instance '{DB_INSTANCE_NAME}'...")

# Delete existing instance if any
try:
    w.database.delete_database_instance(name=DB_INSTANCE_NAME)
    print("Deleted existing instance, waiting...")
    time.sleep(10)
except Exception as e:
    print(f"No existing instance: {e}")

db_instance = w.database.create_database_instance_and_wait(
    database_instance=db_service.DatabaseInstance(
        name=DB_INSTANCE_NAME,
        capacity='CU_2',
    ),
    timeout=datetime.timedelta(seconds=600),
)
print(f"Lakebase instance created: {db_instance.name}, state: {db_instance.state}")
print(f"  Read/write DNS: {db_instance.read_write_dns}")

# Step 5: Create a Database Catalog (Unity Catalog wrapper for Lakebase)
print(f"\nCreating database catalog '{DB_CATALOG_NAME}'...")
try:
    existing = w.database.get_database_catalog(name=DB_CATALOG_NAME)
    print(f"Catalog already exists: {existing}")
except Exception:
    db_catalog = w.database.create_database_catalog(
        catalog=db_service.DatabaseCatalog(
            name=DB_CATALOG_NAME,
            database_instance_name=DB_INSTANCE_NAME,
            database_name=DB_NAME,
            create_database_if_not_exists=True,
        )
    )
    print(f"Database catalog created: {db_catalog}")

# Step 6: Create a Synced Database Table
print(f"\nCreating synced database table...")
SYNCED_TABLE_NAME = f'{DB_CATALOG_NAME}.public.{TABLE_NAME}'

try:
    existing = w.database.get_synced_database_table(name=SYNCED_TABLE_NAME)
    print(f"Synced table already exists: {existing.name}")
except Exception:
    synced_table = w.database.create_synced_database_table(
        synced_table=db_service.SyncedDatabaseTable(
            name=SYNCED_TABLE_NAME,
            database_instance_name=DB_INSTANCE_NAME,
            logical_database_name=DB_NAME,
            spec=db_service.SyncedTableSpec(
                source_table_full_name=FULL_TABLE_NAME,
                primary_key_columns=['account_id'],
                scheduling_policy=db_service.SyncedTableSchedulingPolicy.TRIGGERED,
                create_database_objects_if_missing=True,
            )
        )
    )
    print(f"Synced table created: {synced_table}")

# Wait for synced table to be ready
# Online states: SYNCED_TABLE_ONLINE, SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE, etc.
print("Waiting for synced table to be online...")
ONLINE_STATES = {
    db_service.SyncedTableState.SYNCED_TABLE_ONLINE,
    db_service.SyncedTableState.SYNCED_TABLE_ONLINE_CONTINUOUS_UPDATE,
    db_service.SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE,
    db_service.SyncedTableState.SYNCED_TABLE_ONLINE_TRIGGERED_UPDATE,
    db_service.SyncedTableState.SYNCED_TABLE_ONLINE_UPDATING_PIPELINE_RESOURCES,
}
FAILED_STATES = {
    db_service.SyncedTableState.SYNCED_TABLED_OFFLINE,
    db_service.SyncedTableState.SYNCED_TABLE_OFFLINE_FAILED,
    db_service.SyncedTableState.SYNCED_TABLE_ONLINE_PIPELINE_FAILED,
}
max_wait = 600
start = time.time()
while time.time() - start < max_wait:
    try:
        st = w.database.get_synced_database_table(name=SYNCED_TABLE_NAME)
        status = st.data_synchronization_status
        if status and status.detailed_state:
            state = status.detailed_state
            print(f"  Detailed state: {state}, message: {status.message}")
            if state in ONLINE_STATES:
                print("Synced table is online!")
                break
            elif state in FAILED_STATES:
                print(f"Sync FAILED with state: {state}")
                break
        else:
            print(f"  Status: {status}")
    except Exception as e:
        print(f"  Error checking status: {e}")
    time.sleep(15)

# Step 7: Read lookup keys
print("\nReading lookup keys...")
with open('data/lookup_keys.txt', 'r') as f:
    lookup_keys = [line.strip() for line in f if line.strip()]
print(f"Found {len(lookup_keys)} lookup keys")

# Step 8: Query online (Lakebase/synced) table via SQL
# The database catalog allows querying the Lakebase via SQL warehouse
# This routes through the PostgreSQL online store, not the offline Delta store
print(f"\nQuerying synced table via SQL ({SYNCED_TABLE_NAME})...")

# First check if the synced table has data
check_resp = exec_sql(f"SELECT COUNT(*) FROM {SYNCED_TABLE_NAME}")
print(f"Synced table row count: {check_resp.result}")

# Build IN clause for the lookup keys
keys_list = ', '.join([f"'{k}'" for k in lookup_keys])
query_sql = f"""
SELECT account_id, f1, f2, f3, f4
FROM {SYNCED_TABLE_NAME}
WHERE account_id IN ({keys_list})
"""
result_resp = exec_sql(query_sql)

# Parse results
vectors = {}
if result_resp.result and result_resp.result.data_array:
    for row in result_resp.result.data_array:
        acc_id = row[0]
        f1 = float(row[1])
        f2 = float(row[2])
        f3 = float(row[3])
        f4 = float(row[4])
        vectors[acc_id] = [f1, f2, f3, f4]
    print(f"Retrieved {len(vectors)} vectors from online store")
else:
    # Try fetching results in chunks
    print(f"Checking result chunks...")
    print(f"Result: {result_resp.result}")

# Write output
print(f"\nWriting submission/answers.json...")
os.makedirs('submission', exist_ok=True)
output = {'vectors': vectors}
with open('submission/answers.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"Written submission/answers.json with {len(vectors)} entries")

# Show sample
sample_keys = list(vectors.keys())[:3]
for k in sample_keys:
    print(f"  {k}: {vectors[k]}")
