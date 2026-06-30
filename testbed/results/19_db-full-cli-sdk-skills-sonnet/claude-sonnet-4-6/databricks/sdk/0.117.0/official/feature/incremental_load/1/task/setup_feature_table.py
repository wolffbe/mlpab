"""
Set up incremental feature table on Databricks.
"""
import os
import time
import base64
import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as sdk_catalog
from databricks.sdk.service import jobs as sdk_jobs
from databricks.sdk.service.workspace import ImportFormat, Language

# Configuration
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab66d90d
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]   # mlpab66d90d
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
WAREHOUSE_ID = "4dfab06c923fe3cc"

TABLE_NAME = "incremental3526e9"
FULL_TABLE = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"
ONLINE_TABLE_NAME = f"{FULL_TABLE}_online"
VOLUME_NAME = f"{PREFIX}_increments_vol"
JOB_NAME = f"{PREFIX}_incrementaljob3526e9"
USER = "benedict@logicalclocks.com"
NOTEBOOK_PATH = f"/Users/{USER}/{PREFIX}/incremental_ingest"

print(f"Config: catalog={CATALOG}, schema={SCHEMA_NAME}")
print(f"Table: {FULL_TABLE}")
print(f"Online table: {ONLINE_TABLE_NAME}")
print(f"Job: {JOB_NAME}")

w = WorkspaceClient()

# ── Step 1: Create managed volume ─────────────────────────────────────────────
print("\n[1] Creating volume...")
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}"
try:
    vol = w.volumes.create(
        catalog_name=CATALOG,
        schema_name=SCHEMA_NAME,
        name=VOLUME_NAME,
        volume_type=sdk_catalog.VolumeType.MANAGED,
    )
    print(f"    Created volume: {vol.full_name}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"    Volume already exists, continuing")
    else:
        raise

# ── Step 2: Upload CSV files to volume ────────────────────────────────────────
print("\n[2] Uploading CSV files to volume...")
data_dir = "data"
for i in range(1, 7):
    fname = f"increment_{i:02d}.csv"
    local_path = os.path.join(data_dir, fname)
    remote_path = f"{VOLUME_PATH}/{fname}"
    with open(local_path, "rb") as f:
        w.files.upload(remote_path, f, overwrite=True)
    print(f"    Uploaded {fname}")

# ── Helper: Execute SQL via Statement Execution API ───────────────────────────
def run_sql(stmt, max_wait_seconds=300):
    resp = w.statement_execution.execute_statement(
        statement=stmt,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout="50s",
    )
    deadline = time.time() + max_wait_seconds
    while True:
        status = resp.status
        state = str(status.state) if hasattr(status, "state") else str(status)
        if any(s in state for s in ("SUCCEEDED", "FAILED", "CANCELED", "CLOSED")):
            break
        if time.time() > deadline:
            raise TimeoutError(f"SQL timed out after {max_wait_seconds}s")
        time.sleep(3)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if "SUCCEEDED" not in state:
        err = getattr(status, "error", None)
        raise RuntimeError(f"SQL failed [{state}]: {err}\nSQL: {stmt[:300]}")
    return resp

# ── Step 3: Create Delta feature table ───────────────────────────────────────
print("\n[3] Creating Delta feature table...")
run_sql(f"""
CREATE TABLE IF NOT EXISTS {FULL_TABLE} (
    row_id     STRING   NOT NULL,
    account_id STRING,
    event_time BIGINT,
    amount     DOUBLE,
    category   STRING
)
USING DELTA
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
)
""")
print(f"    Table {FULL_TABLE} ready")

# ── Step 4: Load all 6 increments using COPY INTO ────────────────────────────
print("\n[4] Loading all 6 increments via COPY INTO...")
run_sql(f"""
COPY INTO {FULL_TABLE}
FROM '{VOLUME_PATH}/'
FILEFORMAT = CSV
FORMAT_OPTIONS (
    'header'      = 'true',
    'inferSchema' = 'true'
)
COPY_OPTIONS ('force' = 'false')
""", max_wait_seconds=300)
print("    COPY INTO completed")

resp = run_sql(f"SELECT COUNT(*) as cnt FROM {FULL_TABLE}")
if resp.result and resp.result.data_array:
    print(f"    Row count: {resp.result.data_array[0][0]}")

# ── Step 5: Create UC Online Table for low-latency access ─────────────────────
print("\n[5] Creating UC Online Table...")
try:
    online_table = w.online_tables.create_and_wait(
        table=sdk_catalog.OnlineTable(
            name=ONLINE_TABLE_NAME,
            spec=sdk_catalog.OnlineTableSpec(
                source_table_full_name=FULL_TABLE,
                primary_key_columns=["row_id"],
                timeseries_key="event_time",
                run_triggered=sdk_catalog.OnlineTableSpecTriggeredSchedulingPolicy(),
            ),
        ),
        timeout=datetime.timedelta(seconds=1200),
    )
    print(f"    Online table created: {ONLINE_TABLE_NAME}")
    print(f"    Status: {online_table.status}")
except Exception as e:
    if "already exists" in str(e).lower() or "ALREADY_EXISTS" in str(e):
        print(f"    Online table already exists: {ONLINE_TABLE_NAME}")
    else:
        print(f"    Online table note: {e}")

# ── Step 6: Create ingestion notebook ─────────────────────────────────────────
print("\n[6] Creating ingestion notebook...")
notebook_content = f"""# Databricks notebook source
# COMMAND ----------
# Daily incremental ingest into {FULL_TABLE}
spark.sql(\"\"\"
COPY INTO {FULL_TABLE}
FROM '{VOLUME_PATH}/'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
COPY_OPTIONS ('force' = 'false')
\"\"\")
count = spark.sql("SELECT COUNT(*) FROM {FULL_TABLE}").collect()[0][0]
print(f"Ingest complete. Total rows: {{count}}")
"""

encoded = base64.b64encode(notebook_content.encode()).decode()

try:
    w.workspace.mkdirs(path=f"/Users/{USER}/{PREFIX}")
except Exception:
    pass

w.workspace.import_(
    path=NOTEBOOK_PATH,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=encoded,
    overwrite=True,
)
print(f"    Notebook created: {NOTEBOOK_PATH}")

# ── Step 7: Create daily job with schedule ────────────────────────────────────
print("\n[7] Creating daily ingestion job...")
job = w.jobs.create(
    name=JOB_NAME,
    tasks=[
        sdk_jobs.Task(
            task_key="ingest",
            notebook_task=sdk_jobs.NotebookTask(
                notebook_path=NOTEBOOK_PATH,
                warehouse_id=WAREHOUSE_ID,
            ),
        )
    ],
    schedule=sdk_jobs.CronSchedule(
        quartz_cron_expression="0 0 0 * * ?",  # daily at midnight UTC
        timezone_id="UTC",
        pause_status=sdk_jobs.PauseStatus.UNPAUSED,
    ),
)
print(f"    Job created: id={job.job_id}, name={JOB_NAME}")

print("\n=== All steps completed ===")
print(f"Feature table:  {FULL_TABLE}")
print(f"Online table:   {ONLINE_TABLE_NAME}")
print(f"Job name:       {JOB_NAME}")
print(f"Job ID:         {job.job_id}")
