import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    VolumeType, OnlineTable, OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy
)
from databricks.sdk.service.sql import StatementState, Disposition

w = WorkspaceClient()

schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']   # workspace.mlpab35a71c
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']         # mlpab35a71c
catalog_name, schema_name = schema_full.split('.')

T = 1773306000000
TABLE_NAME = "scores4f5893"
FULL_TABLE = f"{schema_full}.{TABLE_NAME}"
VOLUME_NAME = f"{prefix}_vol"
WH_ID = "4dfab06c923fe3cc"

# ── 1. Create volume ──────────────────────────────────────────────────────────
print("Creating volume...")
try:
    w.volumes.create(
        catalog_name=catalog_name,
        schema_name=schema_name,
        name=VOLUME_NAME,
        volume_type=VolumeType.MANAGED
    )
    print(f"  Volume {VOLUME_NAME} created.")
except Exception as e:
    print(f"  Volume: {e}")

# ── 2. Upload CSV to volume ───────────────────────────────────────────────────
csv_volume_path = f"/Volumes/{catalog_name}/{schema_name}/{VOLUME_NAME}/feature_history.csv"
print(f"Uploading CSV to {csv_volume_path}...")
with open("data/feature_history.csv", "rb") as f:
    w.files.upload(csv_volume_path, f, overwrite=True)
print("  Uploaded.")

# ── 3. Helper to run SQL via statement execution ──────────────────────────────
def run_sql(sql, description=""):
    print(f"  SQL: {description or sql[:80]}")
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=WH_ID,
        wait_timeout="50s",
        catalog=catalog_name,
        schema=schema_name
    )
    # Poll if not done
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(5)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed ({resp.status.state}): {resp.status.error}")
    print(f"    -> OK ({resp.status.state})")
    return resp

# ── 4. Create the feature table with CTAS ────────────────────────────────────
print("Creating feature table...")

create_sql = f"""
CREATE OR REPLACE TABLE {FULL_TABLE}
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
) AS
WITH src AS (
  SELECT
    account_id,
    CAST(event_time AS BIGINT) AS event_time,
    CAST(f1 AS DOUBLE) AS f1,
    CAST(f2 AS DOUBLE) AS f2,
    CAST(f3 AS DOUBLE) AS f3
  FROM read_files(
    '{csv_volume_path}',
    format => 'csv',
    header => true
  )
  WHERE CAST(event_time AS BIGINT) <= {T}
),
ranked AS (
  SELECT
    account_id, f1, f2, f3,
    ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) AS rn
  FROM src
)
SELECT
  account_id,
  ROUND(1.0 / (1.0 + EXP(-({-0.9682} * f1 + {-0.0299} * f2 + {1.2708} * f3 + {-0.1715}))), 6) AS score
FROM ranked
WHERE rn = 1
"""

run_sql(create_sql, "CTAS scores4f5893")

# ── 5. Add primary key constraint ────────────────────────────────────────────
print("Adding primary key...")
run_sql(
    f"ALTER TABLE {FULL_TABLE} ALTER COLUMN account_id SET NOT NULL",
    "SET NOT NULL"
)
run_sql(
    f"ALTER TABLE {FULL_TABLE} ADD CONSTRAINT {TABLE_NAME}_pk PRIMARY KEY (account_id)",
    "ADD PRIMARY KEY"
)

# ── 6. Verify row count ───────────────────────────────────────────────────────
print("Verifying row count...")
resp = run_sql(f"SELECT COUNT(*) AS n FROM {FULL_TABLE}", "COUNT")
if resp.result and resp.result.data_array:
    print(f"  Rows in table: {resp.result.data_array[0][0]}")

# ── 7. Create online table for low-latency access ────────────────────────────
print("Creating online table...")
online_table_name = f"{FULL_TABLE}"  # online table lives alongside the source table
try:
    ot = w.online_tables.create_and_wait(
        table=OnlineTable(
            name=online_table_name,
            spec=OnlineTableSpec(
                source_table_full_name=FULL_TABLE,
                primary_key_columns=["account_id"],
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
            )
        ),
        timeout=1800
    )
    print(f"  Online table ready: {ot.name}")
    print(f"  Serving URL: {ot.table_serving_url}")
except Exception as e:
    print(f"  Online table: {e}")

print("\nDone.")
