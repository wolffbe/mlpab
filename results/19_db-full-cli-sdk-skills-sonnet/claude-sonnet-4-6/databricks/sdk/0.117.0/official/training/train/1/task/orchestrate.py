"""Orchestrate training job on Databricks and create feature table."""
import os
import time
import json
import base64

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
from databricks.sdk.service.catalog import (
    VolumeType,
    OnlineTableSpec,
    OnlineTableSpecContinuousSchedulingPolicy,
)

CATALOG = "workspace"
SCHEMA = "mlpaba7791d"
PREFIX = "mlpaba7791d"
JOB_NAME = "trainjob7b586d"
TABLE_NAME = "predictions7b586d"
VOLUME_NAME = f"{PREFIX}_vol"
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"
FULL_TABLE = f"{FULL_SCHEMA}.{TABLE_NAME}"
ONLINE_TABLE_NAME = f"{FULL_TABLE}_online"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

w = WorkspaceClient()
me = w.current_user.me()
USER = me.user_name
print(f"User: {USER}")

# Step 1: Create volume
print("\n=== Step 1: Create volume ===")
try:
    vol = w.volumes.create(
        catalog_name=CATALOG,
        schema_name=SCHEMA,
        name=VOLUME_NAME,
        volume_type=VolumeType.MANAGED,
    )
    print(f"Created volume: {vol.full_name}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Volume already exists: {VOLUME_PATH}")
    else:
        raise

# Step 2: Upload data files to volume
print("\n=== Step 2: Upload data files ===")
import io
for fname in ["train.csv", "score.csv", "train_model.py"]:
    local_path = f"data/{fname}"
    remote_path = f"{VOLUME_PATH}/{fname}"
    with open(local_path, "rb") as f:
        content = f.read()
    w.files.upload(remote_path, io.BytesIO(content), overwrite=True)
    print(f"  Uploaded {fname}")

# Step 3: Create notebook
print("\n=== Step 3: Create notebook ===")
NOTEBOOK_PATH = f"/Users/{USER}/{PREFIX}/train_model_notebook"

# Create workspace directory
try:
    w.workspace.mkdirs(f"/Users/{USER}/{PREFIX}")
except Exception as e:
    print(f"  mkdirs: {e}")

notebook_source = f"""import os, shutil, importlib.util

# Set up working directory
work_dir = "/tmp/{JOB_NAME}"
os.makedirs(work_dir, exist_ok=True)

# Copy data files from volume to working directory
for fname in ["train.csv", "score.csv", "train_model.py"]:
    shutil.copy(f"{VOLUME_PATH}/{{fname}}", f"{{work_dir}}/{{fname}}")

print("Files in work dir:", os.listdir(work_dir))

# Run train_model.py from the working directory
orig_dir = os.getcwd()
os.chdir(work_dir)

spec = importlib.util.spec_from_file_location("train_model", f"{{work_dir}}/train_model.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.main()

os.chdir(orig_dir)
print("Files after training:", os.listdir(work_dir))

# Read predictions
import pandas as pd
preds = pd.read_csv(f"{{work_dir}}/predictions.csv")
print("Predictions shape:", preds.shape)
print("First rows:", preds.head().to_string())

# Copy predictions CSV to volume for archival
shutil.copy(f"{{work_dir}}/predictions.csv", f"{VOLUME_PATH}/predictions.csv")
print("Copied predictions.csv to volume")

# Write Delta table (the feature table)
spark_df = spark.createDataFrame(preds)
(spark_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("{FULL_TABLE}"))
print("Wrote Delta table: {FULL_TABLE}")

# Enable change data feed for online table sync
spark.sql("ALTER TABLE {FULL_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("Enabled change data feed")

print("DONE")
"""

notebook_encoded = base64.b64encode(notebook_source.encode()).decode()
from databricks.sdk.service.workspace import ImportFormat, Language
w.workspace.import_(
    path=NOTEBOOK_PATH,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=notebook_encoded,
    overwrite=True,
)
print(f"  Created notebook: {NOTEBOOK_PATH}")

# Step 5: Create job with serverless compute
print("\n=== Step 5: Create job ===")
task = jobs.Task(
    task_key="train",
    notebook_task=jobs.NotebookTask(notebook_path=NOTEBOOK_PATH),
)

job_resp = w.jobs.create(
    name=JOB_NAME,
    tasks=[task],
)
job_id = job_resp.job_id
print(f"  Created job: {JOB_NAME} (id={job_id})")

# Step 6: Run job and wait
print("\n=== Step 6: Run job ===")
run_resp = w.jobs.run_now(job_id=job_id)
run_id = run_resp.run_id
print(f"  Run ID: {run_id}")

start = time.time()
max_wait = 1800
while True:
    elapsed = int(time.time() - start)
    run = w.jobs.get_run(run_id=run_id)
    lc = run.state.life_cycle_state if run.state else None
    rs = run.state.result_state if run.state else None
    print(f"  [{elapsed}s] {lc} / {rs}")
    if lc in [
        jobs.RunLifeCycleState.TERMINATED,
        jobs.RunLifeCycleState.SKIPPED,
        jobs.RunLifeCycleState.INTERNAL_ERROR,
    ]:
        break
    if elapsed > max_wait:
        raise TimeoutError("Job timed out")
    time.sleep(30)

final_result = run.state.result_state if run.state else None
print(f"\n  Final result: {final_result}")

if final_result != jobs.RunResultState.SUCCESS:
    try:
        output = w.jobs.get_run_output(run_id=run_id)
        print("Error:", output.error)
        print("Trace:", output.error_trace)
    except Exception as ex:
        print(f"Could not get output: {ex}")
    raise RuntimeError(f"Job failed: {final_result}")

print("  Job completed successfully!")

# Step 7: Create online table
print("\n=== Step 7: Create online table ===")
try:
    online_spec = OnlineTableSpec(
        primary_key_columns=["row_id"],
        source_table_full_name=FULL_TABLE,
        run_continuously=OnlineTableSpecContinuousSchedulingPolicy(),
    )
    ot = w.online_tables.create(
        name=ONLINE_TABLE_NAME,
        spec=online_spec,
    )
    print(f"  Created online table: {ONLINE_TABLE_NAME}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"  Online table already exists: {ONLINE_TABLE_NAME}")
    else:
        print(f"  Online table error: {e}")

print("\n=== All done! ===")
print(f"Job: {JOB_NAME} (id={job_id})")
print(f"Table: {FULL_TABLE}")
print(f"Online table: {ONLINE_TABLE_NAME}")
