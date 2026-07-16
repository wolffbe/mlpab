import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy,
    VolumeType
)

w = WorkspaceClient()

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]   # workspace.mlpab7c79f3
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]    # mlpab7c79f3
catalog, schema_name = SCHEMA.split(".")
TABLE_NAME = "accountse81ff1"
FULL_TABLE_NAME = f"{catalog}.{schema_name}.{TABLE_NAME}"
VOLUME_NAME = f"{PREFIX}_accounts_data"
WAREHOUSE_ID = "4dfab06c923fe3cc"

print(f"catalog={catalog}, schema={schema_name}")
print(f"Full table: {FULL_TABLE_NAME}")
print(f"Volume: {VOLUME_NAME}")

# ── 1. Create a volume to hold the CSVs ──────────────────────────────────────
print("\n[1] Creating volume...")
try:
    vol = w.volumes.create(
        catalog_name=catalog,
        schema_name=schema_name,
        name=VOLUME_NAME,
        volume_type=VolumeType.MANAGED,
    )
    print(f"  Created volume: {vol.full_name}")
except Exception as e:
    print(f"  Volume already exists or error: {e}")

volume_path = f"/Volumes/{catalog}/{schema_name}/{VOLUME_NAME}"
print(f"  Volume path: {volume_path}")

# ── 2. Upload CSV files to the volume ────────────────────────────────────────
print("\n[2] Uploading CSV files...")
for batch_file in ["batch_1.csv", "batch_2.csv", "batch_3.csv"]:
    local_path = f"data/{batch_file}"
    remote_path = f"{volume_path}/{batch_file}"
    with open(local_path, "rb") as f:
        w.files.upload(remote_path, f, overwrite=True)
    print(f"  Uploaded {batch_file} -> {remote_path}")

# ── 3. Execute SQL via warehouse to create + populate the feature table ───────
def run_sql(sql, warehouse_id=WAREHOUSE_ID, description="", timeout=300):
    from databricks.sdk.service.sql import StatementState
    print(f"  SQL: {description or sql[:80]}")
    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout="50s",
    )
    deadline = time.time() + timeout
    while resp.status and resp.status.state in (
        StatementState.PENDING, StatementState.RUNNING
    ):
        if time.time() > deadline:
            raise RuntimeError("SQL timed out")
        time.sleep(3)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status and resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed [{resp.status.state}]: {resp.status.error}")
    print(f"  => {resp.status.state}")
    return resp

print("\n[3] Creating feature table...")
run_sql(f"""
CREATE OR REPLACE TABLE {FULL_TABLE_NAME} (
  row_id    STRING  NOT NULL,
  status    STRING,
  balance   DOUBLE,
  updated_at BIGINT
)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
)
""", description=f"CREATE TABLE {FULL_TABLE_NAME}")

# ── 4. Load batches, keep latest row per row_id ───────────────────────────────
print("\n[4] Loading batches (latest-wins merge)...")
run_sql(f"""
INSERT INTO {FULL_TABLE_NAME}
SELECT row_id, status, CAST(balance AS DOUBLE), CAST(updated_at AS BIGINT)
FROM (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY CAST(updated_at AS BIGINT) DESC) AS rn
  FROM (
    SELECT row_id, status, balance, updated_at
    FROM read_files('{volume_path}/batch_1.csv', format => 'csv', header => true,
                    schema => 'row_id STRING, status STRING, balance STRING, updated_at STRING')
    UNION ALL
    SELECT row_id, status, balance, updated_at
    FROM read_files('{volume_path}/batch_2.csv', format => 'csv', header => true,
                    schema => 'row_id STRING, status STRING, balance STRING, updated_at STRING')
    UNION ALL
    SELECT row_id, status, balance, updated_at
    FROM read_files('{volume_path}/batch_3.csv', format => 'csv', header => true,
                    schema => 'row_id STRING, status STRING, balance STRING, updated_at STRING')
  )
)
WHERE rn = 1
""", description="INSERT latest-wins data from all 3 batches")

# ── 5. Verify the row count ────────────────────────────────────────────────────
print("\n[5] Verifying row count...")
resp = run_sql(f"SELECT COUNT(*) as cnt, COUNT(DISTINCT row_id) as uniq FROM {FULL_TABLE_NAME}",
               description="Count rows")
if resp.result and resp.result.data_array:
    cnt, uniq = resp.result.data_array[0]
    print(f"  Total rows: {cnt}, Unique row_ids: {uniq}")

# ── 6. Add primary key constraint ─────────────────────────────────────────────
print("\n[6] Adding primary key constraint...")
try:
    run_sql(f"""
    ALTER TABLE {FULL_TABLE_NAME}
    ADD CONSTRAINT {TABLE_NAME}_pk PRIMARY KEY (row_id)
    """, description="ADD PRIMARY KEY row_id")
except Exception as e:
    print(f"  PK already exists or not supported: {e}")

# ── 7. Create Online Table for low-latency lookup ─────────────────────────────
online_table_name = f"{FULL_TABLE_NAME}_online"
print(f"\n[7] Creating online table: {online_table_name}...")
try:
    online_table = w.online_tables.create(
        table=OnlineTable(
            name=online_table_name,
            spec=OnlineTableSpec(
                source_table_full_name=FULL_TABLE_NAME,
                primary_key_columns=["row_id"],
                timeseries_key="updated_at",
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
                perform_full_copy=True,
            ),
        )
    )
    print(f"  Online table creation started, waiting for it to become active...")
    # Wait up to 20 minutes for the online table to sync
    result = online_table.result(timeout=1200)
    print(f"  Online table status: {result.status}")
except Exception as e:
    print(f"  Online table creation error: {e}")
    # Try to get existing
    try:
        existing = w.online_tables.get(online_table_name)
        print(f"  Existing online table: {existing.status}")
    except Exception as e2:
        print(f"  Could not retrieve online table: {e2}")

print("\n=== DONE ===")
print(f"Feature table: {FULL_TABLE_NAME}")
print(f"Online table:  {online_table_name}")
