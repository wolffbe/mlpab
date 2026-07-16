"""
Creates a standardized feature table named 'scaled7ecfaf' in Databricks,
then creates an online table from it.
"""
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy,
    VolumeType
)

w = WorkspaceClient()

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]   # workspace.mlpabf0df63
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]    # mlpabf0df63
CATALOG, SCHEMA_NAME = SCHEMA.split(".", 1)

TABLE_NAME = "scaled7ecfaf"
FULL_TABLE = f"{SCHEMA}.{TABLE_NAME}"
WAREHOUSE_ID = "4dfab06c923fe3cc"

print(f"Schema: {SCHEMA}")
print(f"Table: {FULL_TABLE}")

# ── 1. Create a volume to store the raw CSVs ─────────────────────────────────
VOLUME_NAME = "raw_data"
try:
    vol = w.volumes.create(
        catalog_name=CATALOG,
        schema_name=SCHEMA_NAME,
        name=VOLUME_NAME,
        volume_type=VolumeType.MANAGED,
    )
    print(f"Created volume: {vol.full_name}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Volume already exists, using it")
    else:
        raise

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}"

# ── 2. Upload the CSV files to the volume ────────────────────────────────────
data_dir = os.path.join(os.path.dirname(__file__), "data")

for fname in ["features_train.csv", "features_serve.csv"]:
    local_path = os.path.join(data_dir, fname)
    remote_path = f"{VOLUME_PATH}/{fname}"
    with open(local_path, "rb") as f:
        w.files.upload(remote_path, f, overwrite=True)
    print(f"Uploaded {fname} → {remote_path}")

# ── 3. Run SQL on the warehouse to build the standardized table ───────────────

def exec_sql(statement, timeout=300):
    """Execute SQL statement and wait for completion."""
    resp = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout="50s",
    )
    # Poll if not done
    stmt_id = resp.statement_id
    start = time.time()
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if time.time() - start > timeout:
            raise TimeoutError(f"SQL timed out after {timeout}s")
        time.sleep(3)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(
            f"SQL failed ({resp.status.state}): {resp.status.error}"
        )
    return resp

print("\nStep 3: Building standardized feature table via SQL...")

# Drop table if it already exists (idempotent)
exec_sql(f"DROP TABLE IF EXISTS {FULL_TABLE}")
print("  Dropped existing table (if any)")

# Create raw staging tables from CSVs
exec_sql(f"""
CREATE OR REPLACE TABLE {SCHEMA}.raw_train
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
AS SELECT * FROM read_files(
  '{VOLUME_PATH}/features_train.csv',
  format => 'csv',
  header => 'true',
  inferSchema => 'true'
)
""")
print("  Created raw_train table")

exec_sql(f"""
CREATE OR REPLACE TABLE {SCHEMA}.raw_serve
USING DELTA
AS SELECT * FROM read_files(
  '{VOLUME_PATH}/features_serve.csv',
  format => 'csv',
  header => 'true',
  inferSchema => 'true'
)
""")
print("  Created raw_serve table")

# Compute training statistics (population std = stddev_pop)
# Then build the standardized table
exec_sql(f"""
CREATE TABLE {FULL_TABLE}
(
  row_id STRING NOT NULL,
  split  STRING,
  f1     DOUBLE,
  f2     DOUBLE,
  f3     DOUBLE,
  f4     DOUBLE
)
USING DELTA
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
)
""")
print("  Created empty scaled7ecfaf table")

# Insert standardized data (train + serve) using training stats
exec_sql(f"""
INSERT INTO {FULL_TABLE}
WITH train_stats AS (
  SELECT
    avg(f1) AS mean_f1, stddev_pop(f1) AS std_f1,
    avg(f2) AS mean_f2, stddev_pop(f2) AS std_f2,
    avg(f3) AS mean_f3, stddev_pop(f3) AS std_f3,
    avg(f4) AS mean_f4, stddev_pop(f4) AS std_f4
  FROM {SCHEMA}.raw_train
),
combined AS (
  SELECT row_id, 'train' AS split, f1, f2, f3, f4 FROM {SCHEMA}.raw_train
  UNION ALL
  SELECT row_id, 'serve' AS split, f1, f2, f3, f4 FROM {SCHEMA}.raw_serve
)
SELECT
  c.row_id,
  c.split,
  ROUND((c.f1 - s.mean_f1) / s.std_f1, 6) AS f1,
  ROUND((c.f2 - s.mean_f2) / s.std_f2, 6) AS f2,
  ROUND((c.f3 - s.mean_f3) / s.std_f3, 6) AS f3,
  ROUND((c.f4 - s.mean_f4) / s.std_f4, 6) AS f4
FROM combined c
CROSS JOIN train_stats s
ORDER BY c.row_id
""")
print("  Inserted standardized data into scaled7ecfaf")

# Verify row counts
resp = exec_sql(f"""
SELECT split, COUNT(*) as cnt FROM {FULL_TABLE} GROUP BY split ORDER BY split
""")
print(f"  Row counts: {resp.result}")

resp2 = exec_sql(f"SELECT * FROM {FULL_TABLE} LIMIT 3")
print(f"  Sample rows: {resp2.result}")

# ── 4. Create the Online Table ────────────────────────────────────────────────
print("\nStep 4: Creating online table...")

ONLINE_TABLE_NAME = f"{FULL_TABLE}"

try:
    ot = w.online_tables.create(
        table=OnlineTable(
            name=ONLINE_TABLE_NAME,
            spec=OnlineTableSpec(
                source_table_full_name=FULL_TABLE,
                primary_key_columns=["row_id"],
                perform_full_copy=True,
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
            ),
        )
    ).result(timeout=600)
    print(f"Online table created: {ot.name}")
    print(f"  Status: {ot.status}")
except Exception as e:
    print(f"Online table error: {e}")
    # Try to get status if it already exists
    try:
        ot = w.online_tables.get(ONLINE_TABLE_NAME)
        print(f"Online table already exists: {ot.name}, status={ot.status}")
    except Exception as e2:
        print(f"Could not get online table status: {e2}")

print("\nDone!")
