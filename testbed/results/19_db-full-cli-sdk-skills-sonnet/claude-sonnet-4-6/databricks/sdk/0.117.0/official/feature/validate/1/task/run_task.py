import csv
import json
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as cat_svc
from databricks.sdk.service.sql import StatementState

# ── Config ────────────────────────────────────────────────────────────────────
SCHEMA     = os.environ["MLPAB_DATABRICKS_SCHEMA"]   # workspace.mlpabdbae68
PREFIX     = os.environ["MLPAB_DATABRICKS_PREFIX"]   # mlpabdbae68
CATALOG_NAME, SCHEMA_NAME = SCHEMA.split(".", 1)
TABLE_NAME  = "eventsd693d3"
FULL_TABLE  = f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}"
DATA_FILE   = "data/events.csv"
VALID_CATS  = {"grocery", "travel", "salary", "rent", "other"}
WH_ID       = "4dfab06c923fe3cc"

# ── 1. Identify rejected rows (pure stdlib) ───────────────────────────────────
print("Step 1: Classifying rows…")
rejected_ids = []
valid_count  = 0
with open(DATA_FILE, newline="") as f:
    for row in csv.DictReader(f):
        amt_raw  = row.get("amount", "").strip()
        category = row.get("category", "").strip()
        ok = True
        if not amt_raw:
            ok = False
        else:
            try:
                a = float(amt_raw)
                if not (0 <= a <= 10000):
                    ok = False
            except ValueError:
                ok = False
        if category not in VALID_CATS:
            ok = False
        if ok:
            valid_count += 1
        else:
            rejected_ids.append(row["row_id"])

print(f"  Valid: {valid_count}, Rejected: {len(rejected_ids)}")

# ── 2. Write answers.json ─────────────────────────────────────────────────────
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"rejected": rejected_ids}, f, indent=2)
print("  Written submission/answers.json")

# ── 3. Connect ────────────────────────────────────────────────────────────────
print("Step 3: Connecting to Databricks…")
w = WorkspaceClient()
user = w.current_user.me().user_name
print(f"  User: {user}")

# ── 4. Ensure volume exists and upload CSV ───────────────────────────────────
print("Step 4: Uploading CSV to volume…")
volume_name = "task_data"
full_volume  = f"{CATALOG_NAME}.{SCHEMA_NAME}.{volume_name}"
try:
    w.volumes.read(full_volume)
    print(f"  Volume exists: {full_volume}")
except Exception:
    print(f"  Creating volume: {full_volume}")
    w.volumes.create(
        catalog_name=CATALOG_NAME,
        schema_name=SCHEMA_NAME,
        name=volume_name,
        volume_type=cat_svc.VolumeType.MANAGED,
    )

volume_path = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{volume_name}/events_{PREFIX}.csv"
with open(DATA_FILE, "rb") as f:
    w.files.upload(volume_path, f, overwrite=True)
print(f"  Uploaded to {volume_path}")

# ── 5. Helper: run SQL and wait ───────────────────────────────────────────────
def run_sql(sql, description=""):
    print(f"  SQL: {description or sql[:80]}")
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=WH_ID,
        catalog=CATALOG_NAME,
        schema=SCHEMA_NAME,
        wait_timeout="50s",
    )
    stmt_id = resp.statement_id
    # Poll if still running
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(5)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed ({resp.status.state}): {resp.status.error}")
    return resp

# ── 6. Create feature table via SQL ──────────────────────────────────────────
print("Step 6: Creating feature table with filtered data…")

# Drop if exists (clean start)
try:
    run_sql(f"DROP TABLE IF EXISTS {FULL_TABLE}", "drop existing table")
except Exception as e:
    print(f"  (drop warning: {e})")

create_sql = f"""
CREATE TABLE {FULL_TABLE}
USING DELTA
AS
SELECT
  CAST(row_id      AS STRING) AS row_id,
  CAST(account_id  AS STRING) AS account_id,
  CAST(event_time  AS BIGINT) AS event_time,
  CAST(amount      AS DOUBLE) AS amount,
  CAST(category    AS STRING) AS category
FROM read_files(
  '{volume_path}',
  format => 'csv',
  header => true,
  inferSchema => true
)
WHERE
  amount IS NOT NULL
  AND CAST(amount AS DOUBLE) >= 0
  AND CAST(amount AS DOUBLE) <= 10000
  AND category IN ('grocery','travel','salary','rent','other')
"""
run_sql(create_sql, f"CREATE TABLE {FULL_TABLE}")

# Verify row count
cnt_resp = run_sql(f"SELECT COUNT(*) AS n FROM {FULL_TABLE}", "count rows")
rows = cnt_resp.result.data_array if cnt_resp.result else []
print(f"  Table has {rows[0][0] if rows else '?'} rows")

# ── 7. Tag table for feature store metadata ───────────────────────────────────
print("Step 7: Tagging table as feature table…")
tag_sql = f"""
ALTER TABLE {FULL_TABLE}
SET TBLPROPERTIES (
  'feature_store.record_key' = 'row_id',
  'feature_store.event_time' = 'event_time',
  'feature_store.version'    = '1'
)
"""
try:
    run_sql(tag_sql, "set feature table properties")
    print("  Table tagged")
except Exception as e:
    print(f"  (tagging warning: {e})")

# ── 8. Create online table for real-time access ───────────────────────────────
print("Step 8: Creating online table…")
online_name = f"{FULL_TABLE}_online"
try:
    existing = w.online_tables.get(online_name)
    print(f"  Online table already exists: {existing.name}")
except Exception:
    try:
        ot = w.online_tables.create(
            table=cat_svc.OnlineTable(
                name=online_name,
                spec=cat_svc.OnlineTableSpec(
                    source_table_full_name=FULL_TABLE,
                    primary_key_columns=["row_id"],
                    timeseries_key="event_time",
                    run_triggered=cat_svc.OnlineTableSpecTriggeredSchedulingPolicy(),
                ),
            )
        )
        print(f"  Created online table: {online_name}")
        print(f"  Status: {ot}")
    except Exception as e:
        print(f"  Online table creation result: {e}")

# ── 9. Summary ────────────────────────────────────────────────────────────────
print("\n=== DONE ===")
print(f"Feature table : {FULL_TABLE}")
print(f"Online table  : {online_name}")
print(f"Valid rows    : {valid_count}")
print(f"Rejected rows : {len(rejected_ids)}")
print(f"answers.json  : submission/answers.json")
