import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    VolumeType,
)

w = WorkspaceClient()

CATALOG = "workspace"
SCHEMA = "mlpaba00c82"
TABLE_NAME = "transactions9dd1da"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
VOLUME_NAME = "ingest_vol"
VOLUME_FULL = f"{CATALOG}.{SCHEMA}.{VOLUME_NAME}"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
DATA_DIR = "./data"

# ── Step 1: create volume ─────────────────────────────────────────────────────
print("Creating volume …")
try:
    w.volumes.create(
        catalog_name=CATALOG,
        schema_name=SCHEMA,
        name=VOLUME_NAME,
        volume_type=VolumeType.MANAGED,
    )
    print("  Volume created.")
except Exception as exc:
    print(f"  Volume already exists or error (continuing): {exc}")

# ── Step 2: upload CSV files ──────────────────────────────────────────────────
print("Uploading CSV files …")
for fname in ["transactions_export_1.csv", "transactions_export_2.csv"]:
    fpath = os.path.join(DATA_DIR, fname)
    dest = f"{VOLUME_PATH}/{fname}"
    with open(fpath, "rb") as fh:
        w.files.upload(file_path=dest, contents=fh, overwrite=True)
    print(f"  Uploaded {fname} → {dest}")

# ── Step 3: find a warehouse ──────────────────────────────────────────────────
print("Looking for a warehouse …")
warehouses = list(w.warehouses.list())
if not warehouses:
    raise RuntimeError("No SQL warehouses available")
wh = warehouses[0]
warehouse_id = wh.id
print(f"  Using warehouse: {wh.name} ({warehouse_id})")


def run_sql(statement: str, wait_seconds: int = 300) -> None:
    """Execute a SQL statement and wait for it to finish."""
    from databricks.sdk.service import sql as dbsql

    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        wait_timeout="50s",
        on_wait_timeout=dbsql.ExecuteStatementRequestOnWaitTimeout.CONTINUE,
    )
    stmt_id = resp.statement_id
    state = resp.status.state if resp.status else None

    deadline = time.time() + wait_seconds
    while state in (
        dbsql.StatementState.PENDING,
        dbsql.StatementState.RUNNING,
        None,
    ):
        if time.time() > deadline:
            raise TimeoutError(f"Statement {stmt_id} timed out")
        time.sleep(3)
        resp = w.statement_execution.get_statement(stmt_id)
        state = resp.status.state

    if state != dbsql.StatementState.SUCCEEDED:
        raise RuntimeError(
            f"Statement {stmt_id} failed with state={state}\n"
            f"Error: {resp.status.error}"
        )
    print(f"  SQL OK (state={state})")


# ── Step 4: create feature table ─────────────────────────────────────────────
print("Creating feature table …")
create_sql = f"""
CREATE TABLE IF NOT EXISTS {FULL_TABLE} (
  row_id    STRING  NOT NULL,
  account_id STRING,
  event_time BIGINT,
  amount    DOUBLE,
  category  STRING,
  CONSTRAINT {TABLE_NAME}_pk PRIMARY KEY (row_id)
)
TBLPROPERTIES (
  'delta.enableChangeDataFeed'                = 'true',
  'databricks.feature_store.primary_keys'    = 'row_id',
  'databricks.feature_store.event_time_column' = 'event_time'
)
"""
run_sql(create_sql, wait_seconds=120)
print("  Feature table ready.")

# ── Step 5: load data (deduplicated) ─────────────────────────────────────────
print("Loading data with deduplication …")
load_sql = f"""
MERGE INTO {FULL_TABLE} AS tgt
USING (
  SELECT row_id, account_id, event_time, amount, category
  FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY event_time DESC) AS rn
    FROM   read_files(
             '{VOLUME_PATH}/',
             format  => 'csv',
             header  => true,
             schema  => 'row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING'
           )
  )
  WHERE rn = 1
) AS src
ON tgt.row_id = src.row_id
WHEN NOT MATCHED THEN
  INSERT (row_id, account_id, event_time, amount, category)
  VALUES (src.row_id, src.account_id, src.event_time, src.amount, src.category)
"""
run_sql(load_sql, wait_seconds=300)
print("  Data loaded.")

# ── Step 6: verify row count ──────────────────────────────────────────────────
print("Verifying row count …")
from databricks.sdk.service import sql as dbsql

resp = w.statement_execution.execute_statement(
    warehouse_id=warehouse_id,
    statement=f"SELECT COUNT(*) AS n FROM {FULL_TABLE}",
    wait_timeout="30s",
    on_wait_timeout=dbsql.ExecuteStatementRequestOnWaitTimeout.CONTINUE,
)
# wait
stmt_id = resp.statement_id
state = resp.status.state if resp.status else None
deadline = time.time() + 60
while state in (dbsql.StatementState.PENDING, dbsql.StatementState.RUNNING, None):
    if time.time() > deadline:
        break
    time.sleep(2)
    resp = w.statement_execution.get_statement(stmt_id)
    state = resp.status.state

if resp.result and resp.result.data_array:
    count = resp.result.data_array[0][0]
    print(f"  Row count: {count}")

# ── Step 7: create online table ───────────────────────────────────────────────
print("Creating online table …")
try:
    from databricks.sdk.service.catalog import (
        OnlineTableSpec,
        OnlineTableSpecTriggeredSchedulingPolicy,
    )

    online_table = w.online_tables.create(
        name=FULL_TABLE,
        spec=OnlineTableSpec(
            source_table_full_name=FULL_TABLE,
            primary_key_columns=["row_id"],
            timeseries_key="event_time",
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
        ),
    )
    print(f"  Online table created: {online_table.name}")
    print(f"  Status: {online_table.status}")
except Exception as exc:
    print(f"  Online table creation error: {exc}")
    # Try alternative approach — some SDK versions use different parameters
    try:
        from databricks.sdk.service.catalog import OnlineTable

        ot = w.online_tables.create_and_wait(
            name=FULL_TABLE,
            spec=OnlineTableSpec(
                source_table_full_name=FULL_TABLE,
                primary_key_columns=["row_id"],
                timeseries_key="event_time",
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
            ),
        )
        print(f"  Online table ready: {ot.name}")
    except Exception as exc2:
        print(f"  Second attempt error: {exc2}")

print("\nDone.")
