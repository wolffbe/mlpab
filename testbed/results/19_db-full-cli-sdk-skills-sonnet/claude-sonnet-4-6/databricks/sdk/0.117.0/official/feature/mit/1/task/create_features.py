"""
Create the featuresb1ea93 feature table on Databricks.
"""
import io
import os
import time
import base64

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as ws_svc
from databricks.sdk.service.jobs import (
    SubmitTask, NotebookTask as NBTask,
)

w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']   # workspace.mlpab17de0a
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']    # mlpab17de0a
user = w.current_user.me().user_name

catalog, schema_name = schema.split('.')
notebook_dir = f"/Users/{user}/{prefix}"

print(f"Schema: {schema}")
print(f"Prefix: {prefix}")
print(f"Notebook dir: {notebook_dir}")
print(f"User: {user}")

# ── 1. Create a volume for the data files ──────────────────────────────────────
print("\n1. Creating volume...")
try:
    from databricks.sdk.service.catalog import VolumeType
    vol = w.volumes.create(
        catalog_name=catalog,
        schema_name=schema_name,
        name="data_upload",
        volume_type=VolumeType.MANAGED,
    )
    print(f"   Volume created: {vol.full_name}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"   Volume already exists")
    else:
        raise

volume_files_path = f"/Volumes/{catalog}/{schema_name}/data_upload"

# ── 2. Upload the CSV files to the volume ─────────────────────────────────────
print("\n2. Uploading CSV files to volume...")

for fname in ["transactions.csv", "fx_rates.csv"]:
    local_path = f"data/{fname}"
    remote_path = f"{volume_files_path}/{fname}"
    with open(local_path, "rb") as f:
        data = f.read()
    w.files.upload(
        file_path=remote_path,
        contents=io.BytesIO(data),
        overwrite=True,
    )
    print(f"   Uploaded {fname} -> {remote_path}")

# ── 3. Create the computation notebook ────────────────────────────────────────
print("\n3. Creating notebook...")

try:
    w.workspace.mkdirs(notebook_dir)
except Exception:
    pass

notebook_path = f"{notebook_dir}/create_featuresb1ea93"

notebook_content = f'''# Databricks notebook source
# MAGIC %md
# MAGIC # Create featuresb1ea93 feature table

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from databricks.feature_engineering import FeatureEngineeringClient

# COMMAND ----------

catalog = "{catalog}"
schema_name = "{schema_name}"
volume_path = "{volume_files_path}"

# COMMAND ----------

# Read transactions
txn = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{{volume_path}}/transactions.csv")
print("Transactions schema:")
txn.printSchema()
print(f"Transactions count: {{txn.count()}}")

# Read fx rates
fx = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{{volume_path}}/fx_rates.csv")
print("FX rates:")
fx.show()

# COMMAND ----------

# Join to get fx_rate per row
df = txn.join(fx, on="currency", how="left")

# amount_usd = amount * fx_rate
df = df.withColumn("amount_usd", (F.col("amount") * F.col("fx_rate")).cast("double"))

# is_weekend: event_time is in epoch milliseconds
# Convert ms -> seconds for timestamp, then check day of week (UTC)
# dayofweek in Spark: 1=Sunday, 2=Monday, ..., 7=Saturday
df = df.withColumn(
    "_ts_utc", F.to_utc_timestamp(
        (F.col("event_time") / 1000).cast("timestamp"), "UTC"
    )
)
df = df.withColumn(
    "day_of_week", F.dayofweek(F.col("_ts_utc"))
)
# Sunday=1, Saturday=7
df = df.withColumn(
    "is_weekend",
    F.when((F.col("day_of_week") == 1) | (F.col("day_of_week") == 7), 1).otherwise(0).cast("int")
)

# COMMAND ----------

# amount_7d: rolling 7-day sum of amount per account
# event_time is in milliseconds; 7 days = 7 * 24 * 60 * 60 * 1000 = 604800000 ms
window_7d = (
    Window.partitionBy("account_id")
    .orderBy(F.col("event_time").cast("long"))
    .rangeBetween(-604800000, 0)
)
df = df.withColumn("amount_7d", F.sum("amount").over(window_7d).cast("double"))

# COMMAND ----------

# Select final columns
feature_df = df.select(
    F.col("row_id"),
    F.col("account_id"),
    F.col("event_time").cast("long").alias("event_time"),
    F.col("amount_usd"),
    F.col("is_weekend"),
    F.col("amount_7d"),
)

print("Feature dataframe preview:")
feature_df.show(10)
feature_df.printSchema()
print(f"Total rows: {{feature_df.count()}}")

# COMMAND ----------

# Write as Feature Engineering table
fe = FeatureEngineeringClient()

table_name = f"{{catalog}}.{{schema_name}}.featuresb1ea93"
print(f"Creating feature table: {{table_name}}")

fe.create_table(
    name=table_name,
    primary_keys=["row_id"],
    timestamp_keys=["event_time"],
    df=feature_df,
    description="Transaction features: amount_usd, is_weekend, amount_7d",
)
print("Feature table created successfully!")

# COMMAND ----------

# Verify the table
verify_df = spark.table(table_name)
print(f"Verification - Row count: {{verify_df.count()}}")
verify_df.show(5)
print("SUCCESS: Feature table featuresb1ea93 is ready.")
'''

content_b64 = base64.b64encode(notebook_content.encode()).decode()

w.workspace.import_(
    path=notebook_path,
    format=ws_svc.ExportFormat.SOURCE,
    language=ws_svc.Language.PYTHON,
    content=content_b64,
    overwrite=True,
)
print(f"   Notebook created: {notebook_path}")

# ── 4. Submit the notebook as a one-time run ───────────────────────────────────
print("\n4. Submitting notebook run...")

run_resp = w.jobs.submit(
    run_name=f"{prefix}_create_featuresb1ea93",
    tasks=[
        SubmitTask(
            task_key="create_features",
            notebook_task=NBTask(
                notebook_path=notebook_path,
            ),
        )
    ],
)

run_id = run_resp.run_id
print(f"   Run submitted: run_id={run_id}")
run_url = f"{w.config.host}/#job/runs/{run_id}"
print(f"   Run URL: {run_url}")

# ── 5. Wait for completion ─────────────────────────────────────────────────────
print("\n5. Waiting for notebook run to complete...")
max_wait = 1800  # 30 minutes
start_time = time.time()
prev_state = None

while time.time() - start_time < max_wait:
    run_state = w.jobs.get_run(run_id=run_id)
    state = run_state.state
    life_cycle = state.life_cycle_state.value if state and state.life_cycle_state else "UNKNOWN"
    result = state.result_state.value if state and state.result_state else ""
    current = f"{life_cycle}/{result}"

    if current != prev_state:
        elapsed = int(time.time() - start_time)
        print(f"   [{elapsed}s] Status: {current}")
        prev_state = current

    if life_cycle in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
        break
    time.sleep(15)

# Check result
state = w.jobs.get_run(run_id=run_id).state
life_cycle = state.life_cycle_state.value if state and state.life_cycle_state else "UNKNOWN"
result = state.result_state.value if state and state.result_state else "UNKNOWN"

if result != "SUCCESS":
    # Try to get error output
    try:
        for task_run in w.jobs.get_run(run_id=run_id).tasks or []:
            task_run_id = task_run.run_id
            if task_run_id:
                out = w.jobs.get_run_output(run_id=task_run_id)
                if out.error:
                    print(f"   Task error: {out.error}")
                if out.error_trace:
                    print(f"   Error trace: {out.error_trace[:2000]}")
    except Exception as e2:
        print(f"   Could not get output: {e2}")
    raise RuntimeError(f"Notebook run failed: lifecycle={life_cycle}, result={result}")

print(f"\n   Notebook run completed successfully! (lifecycle={life_cycle}, result={result})")

# ── 6. Create Online Table for low-latency access ─────────────────────────────
print("\n6. Creating online table for low-latency access...")

from databricks.sdk.service.catalog import (
    OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
)

table_name = f"{catalog}.{schema_name}.featuresb1ea93"
online_table_name = f"{catalog}.{schema_name}.featuresb1ea93_online"

try:
    ot = w.online_tables.create(
        name=online_table_name,
        spec=OnlineTableSpec(
            source_table_full_name=table_name,
            primary_key_columns=["row_id"],
            timeseries_key="event_time",
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
        ),
    )
    print(f"   Online table creation initiated: {online_table_name}")

    # Wait for online table to become active
    print("   Waiting for online table to sync...")
    max_wait_ot = 600
    start_ot = time.time()
    while time.time() - start_ot < max_wait_ot:
        try:
            ot_info = w.online_tables.get(name=online_table_name)
            status = ot_info.status
            if status:
                ds = str(status.detailed_state) if hasattr(status, 'detailed_state') else str(status)
                print(f"   Online table status: {ds}")
                if any(x in ds.upper() for x in ['ONLINE', 'ACTIVE', 'SYNCED', 'PROVISIONED']):
                    print("   Online table is ready!")
                    break
                if any(x in ds.upper() for x in ['FAIL', 'ERROR', 'OFFLINE_FAILED']):
                    print(f"   Online table issue: {ds}")
                    break
        except Exception as e_ot:
            print(f"   Status check error: {e_ot}")
        time.sleep(20)

except Exception as e:
    print(f"   Online table error: {e}")
    if "already exists" in str(e).lower():
        print("   (online table already exists - OK)")

print("\nAll done! Feature table featuresb1ea93 is ready on the platform.")
print(f"Table: {catalog}.{schema_name}.featuresb1ea93")
