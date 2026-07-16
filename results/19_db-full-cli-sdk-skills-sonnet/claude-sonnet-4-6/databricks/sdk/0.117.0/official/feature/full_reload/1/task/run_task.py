import os
import csv
import time
import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState, Disposition
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

SCHEMA = os.environ['MLPAB_DATABRICKS_SCHEMA']   # workspace.mlpab229d43
catalog, db = SCHEMA.split('.', 1)
WAREHOUSE_ID = '4dfab06c923fe3cc'
TABLE = 'customersc31b07'
FULL_TABLE = f'{SCHEMA}.{TABLE}'

w = WorkspaceClient()


def exec_sql(sql):
    r = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout='50s',
        disposition=Disposition.INLINE,
    )
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state == StatementState.FAILED:
        raise Exception(f'SQL failed: {r.status.error.message}\nSQL: {sql[:300]}')
    return r


def escape_str(s):
    return s.replace("'", "''")


# ── Step 1: Create v1 feature table ──────────────────────────────────────────
print('Creating v1 feature table ...')
exec_sql(f"""
CREATE OR REPLACE TABLE {FULL_TABLE} (
    row_id      STRING NOT NULL,
    name        STRING,
    balance_eur DOUBLE,
    updated_at  BIGINT,
    CONSTRAINT {TABLE}_pk PRIMARY KEY (row_id)
)
USING DELTA
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
)
""")
print('v1 table created')

# ── Step 2: Load initial data ─────────────────────────────────────────────────
print('Loading initial data ...')
with open('data/initial_export.csv') as f:
    v1_rows = list(csv.DictReader(f))

batch_size = 100
for i in range(0, len(v1_rows), batch_size):
    batch = v1_rows[i:i + batch_size]
    vals = ', '.join(
        f"('{escape_str(r['row_id'])}', '{escape_str(r['name'])}', "
        f"{r['balance_eur']}, {r['updated_at']})"
        for r in batch
    )
    exec_sql(f"INSERT INTO {FULL_TABLE} VALUES {vals}")

print(f'Loaded {len(v1_rows)} rows into v1')

# ── Step 3: Drop v1 and create v2 ────────────────────────────────────────────
print('Dropping v1, creating v2 ...')
exec_sql(f"DROP TABLE IF EXISTS {FULL_TABLE}")

exec_sql(f"""
CREATE TABLE {FULL_TABLE} (
    row_id    STRING NOT NULL,
    full_name STRING,
    balance   DOUBLE,
    currency  STRING,
    updated_at BIGINT,
    CONSTRAINT {TABLE}_pk PRIMARY KEY (row_id)
)
USING DELTA
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
)
""")
print('v2 table created')

# ── Step 4: Load new (v2) data ────────────────────────────────────────────────
print('Loading new data ...')
with open('data/reload/new_export.csv') as f:
    v2_rows = list(csv.DictReader(f))

for i in range(0, len(v2_rows), batch_size):
    batch = v2_rows[i:i + batch_size]
    vals = ', '.join(
        f"('{escape_str(r['row_id'])}', '{escape_str(r['full_name'])}', "
        f"{r['balance']}, '{escape_str(r['currency'])}', {r['updated_at']})"
        for r in batch
    )
    exec_sql(f"INSERT INTO {FULL_TABLE} VALUES {vals}")

print(f'Loaded {len(v2_rows)} rows into v2')

# Verify row count
result = exec_sql(f'SELECT COUNT(*) FROM {FULL_TABLE}')
count = result.result.data_array[0][0]
print(f'Row count in {FULL_TABLE}: {count}')

# ── Step 5: Create Online Table for low-latency access ────────────────────────
print('Creating online table ...')

# Delete existing online table if present
try:
    w.online_tables.delete(FULL_TABLE)
    print('Deleted existing online table, waiting before re-creating ...')
    time.sleep(10)
except Exception as e:
    print(f'(no existing online table to delete: {e})')

ot = w.online_tables.create_and_wait(
    table=OnlineTable(
        name=FULL_TABLE,
        spec=OnlineTableSpec(
            source_table_full_name=FULL_TABLE,
            primary_key_columns=['row_id'],
            timeseries_key='updated_at',
            perform_full_copy=True,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
        ),
    ),
    timeout=datetime.timedelta(minutes=25),
)

print(f'Online table ready: {ot.name}')
print(f'  status: {ot.status}')
print('All done.')
