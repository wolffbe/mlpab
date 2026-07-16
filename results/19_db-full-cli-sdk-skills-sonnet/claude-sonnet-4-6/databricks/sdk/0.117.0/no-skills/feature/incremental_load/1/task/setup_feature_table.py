"""
Sets up incremental3526e9 feature table, loads all 6 CSV increments,
creates an online table for real-time access, and creates a recurring job.
"""
import os
import base64
import json
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType, OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
from databricks.sdk.service import jobs as jobs_svc
from databricks.sdk.service.workspace import ImportFormat, Language
from databricks.sdk.service.compute import DataSecurityMode, ClusterSpec as ComputeClusterSpec, Environment as ComputeEnvironment
import databricks.sdk.service.database as db_svc

w = WorkspaceClient()

# Environment config
CATALOG = "workspace"
SCHEMA = "mlpab9db404"
PREFIX = "mlpab9db404"
TABLE_NAME = "incremental3526e9"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
VOLUME_NAME = "incremental_data"
FULL_VOLUME = f"{CATALOG}.{SCHEMA}.{VOLUME_NAME}"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
JOB_NAME = f"{PREFIX}_incrementaljob3526e9"
WH_ID = "4dfab06c923fe3cc"
WAREHOUSE_ID = WH_ID

me = w.current_user.me()
MY_USER = me.user_name
NOTEBOOK_PATH = f"/Users/{MY_USER}/{PREFIX}/ingest_incremental"

print(f"Catalog: {CATALOG}, Schema: {SCHEMA}")
print(f"Table: {FULL_TABLE}")
print(f"Volume: {FULL_VOLUME}")
print(f"Job: {JOB_NAME}")
print(f"User: {MY_USER}")


def run_sql(statement, timeout="30s"):
    """Run SQL and wait for result."""
    resp = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout=timeout,
        catalog=CATALOG,
        schema=SCHEMA,
    )
    # Poll if not done
    stmt_id = resp.statement_id
    while resp.status and resp.status.state and resp.status.state.value in ("PENDING", "RUNNING"):
        time.sleep(2)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status and resp.status.state and resp.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"SQL failed: {resp.status.error}")
    return resp


# Step 1: Create volume for CSV storage
print("\n=== Step 1: Creating volume ===")
try:
    vol = w.volumes.create(
        catalog_name=CATALOG,
        schema_name=SCHEMA,
        name=VOLUME_NAME,
        volume_type=VolumeType.MANAGED,
    )
    print(f"Volume created: {vol.full_name}")
except Exception as e:
    if "already exists" in str(e).lower() or "ALREADY_EXISTS" in str(e):
        print(f"Volume already exists, continuing")
    else:
        raise


# Step 2: Upload CSV files to volume
print("\n=== Step 2: Uploading CSV files to volume ===")
for i in range(1, 7):
    fname = f"increment_0{i}.csv"
    local_path = f"data/{fname}"
    remote_path = f"{VOLUME_PATH}/{fname}"
    with open(local_path, "rb") as f:
        w.files.upload(remote_path, f, overwrite=True)
    print(f"Uploaded {fname} -> {remote_path}")


# Step 3: Create the Delta feature table
print("\n=== Step 3: Creating feature table ===")
create_sql = f"""
CREATE TABLE IF NOT EXISTS {FULL_TABLE} (
    row_id     STRING NOT NULL,
    account_id STRING,
    event_time BIGINT,
    amount     DOUBLE,
    category   STRING
)
USING DELTA
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
)
"""
run_sql(create_sql)
print(f"Table created: {FULL_TABLE}")


# Step 4: Load all 6 CSV increments into the table
print("\n=== Step 4: Loading all CSV increments ===")
for i in range(1, 7):
    fname = f"increment_0{i}.csv"
    remote_path = f"{VOLUME_PATH}/{fname}"
    load_sql = f"""
    COPY INTO {FULL_TABLE}
    FROM '{remote_path}'
    FILEFORMAT = CSV
    FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
    """
    run_sql(load_sql)
    print(f"Loaded {fname}")


# Verify row count
count_resp = run_sql(f"SELECT COUNT(*) as cnt FROM {FULL_TABLE}")
rows = count_resp.result.data_array if count_resp.result else []
print(f"Total rows loaded: {rows[0][0] if rows else 'unknown'}")


# Step 5: Create online/synced table for low-latency access
print("\n=== Step 5: Creating online table (Synced Table via Lakebase) ===")
DB_INSTANCE_NAME = f"{PREFIX}-db"
SYNCED_TABLE_FULL = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

# First try old online tables API (may be deprecated)
try:
    online_table_spec = OnlineTable(
        name=SYNCED_TABLE_FULL,
        spec=OnlineTableSpec(
            source_table_full_name=FULL_TABLE,
            primary_key_columns=["row_id"],
            timeseries_key="event_time",
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
            perform_full_copy=True,
        ),
    )
    ot = w.online_tables.create(table=online_table_spec)
    print(f"Online table creation initiated: {ot}")
except Exception as e:
    print(f"Online tables API note: {e}")
    # Try Synced Tables via Lakebase (newer approach)
    print("Trying Synced Tables via Lakebase database instance...")
    try:
        # Create a database instance first (Lakebase)
        existing_instances = list(w.database.list_database_instances())
        existing_names = [inst.name for inst in existing_instances]
        if DB_INSTANCE_NAME not in existing_names:
            print(f"Creating database instance: {DB_INSTANCE_NAME}")
            db_instance = w.database.create_database_instance_and_wait(
                database_instance=db_svc.DatabaseInstance(
                    name=DB_INSTANCE_NAME,
                    capacity="CU_1",
                )
            )
            print(f"Database instance created: {db_instance.name}")
        else:
            print(f"Database instance already exists: {DB_INSTANCE_NAME}")
            db_instance = next(inst for inst in existing_instances if inst.name == DB_INSTANCE_NAME)

        # Create synced table
        synced_table = w.database.create_synced_database_table(
            synced_table=db_svc.SyncedDatabaseTable(
                name=SYNCED_TABLE_FULL,
                database_instance_name=DB_INSTANCE_NAME,
                spec=db_svc.SyncedTableSpec(
                    source_table_full_name=FULL_TABLE,
                    primary_key_columns=["row_id"],
                    timeseries_key="event_time",
                    scheduling_policy=db_svc.SyncedTableSchedulingPolicy.TRIGGERED,
                    create_database_objects_if_missing=True,
                ),
            )
        )
        print(f"Synced table created: {synced_table}")
    except Exception as e2:
        print(f"Synced table creation note: {e2}")


# Step 6: Create notebook for the daily ingestion job
print("\n=== Step 6: Creating ingestion notebook ===")
notebook_content = f"""# Databricks notebook source
# MAGIC %md
# MAGIC # Daily Incremental Load for {TABLE_NAME}

# COMMAND ----------
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

catalog = "{CATALOG}"
schema = "{SCHEMA}"
table_name = "{TABLE_NAME}"
full_table = f"{{catalog}}.{{schema}}.{{table_name}}"
volume_path = "{VOLUME_PATH}"

# List CSV files in volume and load any not yet processed
import subprocess

# Load all CSV files from volume into the table
# In a production setup, you'd track which files have been loaded
# using a checkpoint or watermark

from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

schema_def = StructType([
    StructField("row_id", StringType(), True),
    StructField("account_id", StringType(), True),
    StructField("event_time", LongType(), True),
    StructField("amount", DoubleType(), True),
    StructField("category", StringType(), True),
])

# Use COPY INTO to idempotently load new files
spark.sql(f\"\"\"
COPY INTO {{full_table}}
FROM '{{volume_path}}'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false')
COPY_OPTIONS ('mergeSchema' = 'true')
\"\"\")

print(f"Ingestion complete for {{full_table}}")
row_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {{full_table}}").collect()[0]['cnt']
print(f"Total rows: {{row_count}}")
"""

# Make the notebook directory
try:
    w.workspace.mkdirs(f"/Users/{MY_USER}/{PREFIX}")
    print(f"Created directory /Users/{MY_USER}/{PREFIX}")
except Exception as e:
    print(f"Directory note: {e}")

# Import the notebook
nb_content_encoded = base64.b64encode(notebook_content.encode("utf-8")).decode("utf-8")
try:
    w.workspace.import_(
        path=NOTEBOOK_PATH,
        content=nb_content_encoded,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        overwrite=True,
    )
    print(f"Notebook created: {NOTEBOOK_PATH}")
except Exception as e:
    print(f"Notebook creation error: {e}")
    raise


# Step 7: Create recurring job with daily schedule
print("\n=== Step 7: Creating recurring job ===")

# Check if job already exists
existing_jobs = list(w.jobs.list(name=JOB_NAME))
if existing_jobs:
    print(f"Job already exists with id {existing_jobs[0].job_id}, deleting to recreate")
    w.jobs.delete(existing_jobs[0].job_id)

job = w.jobs.create(
    name=JOB_NAME,
    tasks=[
        jobs_svc.Task(
            task_key="ingest_increment",
            description="Daily incremental load of events data",
            notebook_task=jobs_svc.NotebookTask(
                notebook_path=NOTEBOOK_PATH,
                source=jobs_svc.Source.WORKSPACE,
            ),
            environment_key="default",
        )
    ],
    environments=[
        jobs_svc.JobEnvironment(
            environment_key="default",
            spec=ComputeEnvironment(
                client="1",
            ),
        )
    ],
    schedule=jobs_svc.CronSchedule(
        quartz_cron_expression="0 0 2 * * ?",  # Daily at 2 AM UTC
        timezone_id="UTC",
        pause_status=jobs_svc.PauseStatus.UNPAUSED,
    ),
)
print(f"Job created: {JOB_NAME} (id={job.job_id})")


# Step 8: Write submission/answers.json
print("\n=== Step 8: Writing answers.json ===")
os.makedirs("submission", exist_ok=True)
answers = {"job_name": "incrementaljob3526e9"}
with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)
print(f"Written: submission/answers.json -> {answers}")


print("\n=== DONE ===")
print(f"Feature table: {FULL_TABLE}")
print(f"Job: {JOB_NAME}")
