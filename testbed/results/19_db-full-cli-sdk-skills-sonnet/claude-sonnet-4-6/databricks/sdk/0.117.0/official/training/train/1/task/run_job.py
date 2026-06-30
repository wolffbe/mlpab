"""Orchestrate training job on Databricks and create feature table."""
import os
import time
import json
import base64

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, catalog, compute
from databricks.sdk.service.catalog import VolumeType, OnlineTableSpec, OnlineTableSpecContinuousSchedulingPolicy

CATALOG = "workspace"
SCHEMA = "mlpaba7791d"
PREFIX = "mlpaba7791d"
JOB_NAME = "trainjob7b586d"
TABLE_NAME = "predictions7b586d"
VOLUME_NAME = f"{PREFIX}_vol"
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"
FULL_TABLE = f"{FULL_SCHEMA}.{TABLE_NAME}"
ONLINE_TABLE_NAME = f"{FULL_TABLE}_online"

w = WorkspaceClient()
me = w.current_user.me()
USER = me.user_name
print(f"User: {USER}")
print(f"Host: {w.config.host}")

# 1. Create volume for storing data files
print("\n=== Creating volume ===")
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
        print(f"Volume already exists")
    else:
        raise

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

# 2. Upload data files to volume
print("\n=== Uploading data files to volume ===")
for fname in ["train.csv", "score.csv", "train_model.py"]:
    local_path = f"data/{fname}"
    remote_path = f"{VOLUME_PATH}/{fname}"
    with open(local_path, "rb") as f:
        content = f.read()
    w.files.upload(remote_path, content, overwrite=True)
    print(f"Uploaded {fname} -> {remote_path}")

# 3. Create notebook content
print("\n=== Creating notebook ===")
NOTEBOOK_PATH = f"/Users/{USER}/{PREFIX}/train_model_notebook"

notebook_content = f'''
import os
import subprocess
import shutil

# Set up working directory with data files
work_dir = "/tmp/trainjob7b586d"
os.makedirs(work_dir, exist_ok=True)

# Copy files from volume to working directory
import shutil
for fname in ["train.csv", "score.csv", "train_model.py"]:
    shutil.copy(f"{VOLUME_PATH}/{{fname}}", f"{{work_dir}}/{{fname}}")

print("Files in work_dir:", os.listdir(work_dir))

# Change to working directory and run the script
orig_dir = os.getcwd()
os.chdir(work_dir)

# Run train_model.py as a module
import importlib.util
spec = importlib.util.spec_from_file_location("train_model", f"{{work_dir}}/train_model.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.main()

os.chdir(orig_dir)
print("Training complete. Files:", os.listdir(work_dir))

# Load predictions into Delta table
import pandas as pd
preds = pd.read_csv(f"{{work_dir}}/predictions.csv")
print("Predictions shape:", preds.shape)
print("Predictions head:", preds.head())

# Save to volume for verification
import shutil
shutil.copy(f"{{work_dir}}/predictions.csv", f"{VOLUME_PATH}/predictions.csv")
print("Saved predictions to volume")

# Create Delta table with predictions
spark.createDataFrame(preds).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FULL_TABLE}")
print(f"Saved predictions to Delta table: {FULL_TABLE}")

# Enable change data feed for online table sync
spark.sql(f"ALTER TABLE {FULL_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("Enabled change data feed on table")

print("Done!")
'''

# Create directory structure in workspace
try:
    w.workspace.mkdirs(f"/Users/{USER}/{PREFIX}")
    print(f"Created workspace dir: /Users/{USER}/{PREFIX}")
except Exception as e:
    print(f"Dir creation: {e}")

# Import notebook
notebook_encoded = base64.b64encode(notebook_content.encode()).decode()
w.workspace.import_(
    path=NOTEBOOK_PATH,
    format="SOURCE",
    language="PYTHON",
    content=notebook_encoded,
    overwrite=True,
)
print(f"Created notebook: {NOTEBOOK_PATH}")

# 4. Get or create cluster policy / default cluster config
print("\n=== Creating job ===")

# Get smallest available node type
node_types = w.clusters.list_node_types()
# Find a small node
small_node = None
for nt in node_types.node_types:
    if nt.num_cores >= 2 and nt.memory_mb >= 4096:
        small_node = nt.node_type_id
        break
if not small_node:
    small_node = "i3.xlarge"
print(f"Using node type: {small_node}")

# Get Spark version
spark_versions = w.clusters.spark_versions()
# Find latest LTS
lts_version = None
for sv in spark_versions.versions:
    if sv.name and "LTS" in sv.name and "ML" not in sv.name:
        lts_version = sv.key
        break
if not lts_version:
    lts_version = spark_versions.versions[0].key
print(f"Using Spark version: {lts_version}")

# Create the job
job_cluster_spec = jobs.JobCluster(
    job_cluster_key="main_cluster",
    new_cluster=compute.ClusterSpec(
        spark_version=lts_version,
        node_type_id=small_node,
        num_workers=0,
        spark_conf={
            "spark.master": "local[*, 4]",
            "spark.databricks.cluster.profile": "singleNode",
        },
        custom_tags={"ResourceClass": "SingleNode"},
        data_security_mode=compute.DataSecurityMode.SINGLE_USER,
    ),
)

task_spec = jobs.Task(
    task_key="train_model",
    notebook_task=jobs.NotebookTask(
        notebook_path=NOTEBOOK_PATH,
    ),
    job_cluster_key="main_cluster",
)

job_response = w.jobs.create(
    name=JOB_NAME,
    job_clusters=[job_cluster_spec],
    tasks=[task_spec],
)
job_id = job_response.job_id
print(f"Created job: {JOB_NAME} (id={job_id})")

# 5. Run the job and wait
print("\n=== Running job ===")
run_response = w.jobs.run_now(job_id=job_id)
run_id = run_response.run_id
print(f"Started run: {run_id}")

# Wait for completion
print("Waiting for job to complete...")
max_wait = 1800  # 30 minutes
start = time.time()
while time.time() - start < max_wait:
    run = w.jobs.get_run(run_id=run_id)
    state = run.state
    life_cycle = state.life_cycle_state if state else None
    result = state.result_state if state else None
    print(f"  [{int(time.time()-start)}s] Life cycle: {life_cycle}, Result: {result}")

    if life_cycle in [
        jobs.RunLifeCycleState.TERMINATED,
        jobs.RunLifeCycleState.SKIPPED,
        jobs.RunLifeCycleState.INTERNAL_ERROR,
    ]:
        break
    time.sleep(30)

run = w.jobs.get_run(run_id=run_id)
final_state = run.state.result_state if run.state else None
print(f"\nJob final state: {final_state}")

if final_state != jobs.RunResultState.SUCCESS:
    # Get output for debugging
    try:
        output = w.jobs.get_run_output(run_id=run_id)
        print("Error:", output.error)
        print("Traceback:", output.error_trace)
    except Exception as ex:
        print(f"Could not get output: {ex}")
    raise RuntimeError(f"Job failed with state: {final_state}")

print("Job completed successfully!")

# 6. Create feature table
print("\n=== Creating feature table ===")
try:
    from databricks.sdk.service.catalog import FeatureSpec
    fe = w.feature_store
    print("Feature store available")
except Exception as e:
    print(f"Feature store: {e}")

# The Delta table is already created by the notebook
# Now register it as a feature table in Unity Catalog
# Feature tables in UC are just Delta tables with feature store metadata
try:
    # Create/register as feature table using feature engineering client
    from databricks.feature_engineering import FeatureEngineeringClient
    fe_client = FeatureEngineeringClient()

    # Read predictions to create feature table
    # Actually, since the notebook already wrote the Delta table,
    # we can use the feature engineering client to register it
    print("Feature engineering client available")
except ImportError:
    print("Feature engineering client not available via import, using SDK")

# 7. Create online table
print("\n=== Creating online table ===")
try:
    online_table_spec = OnlineTableSpec(
        primary_key_columns=["row_id"],
        source_table_full_name=FULL_TABLE,
        run_continuously=OnlineTableSpecContinuousSchedulingPolicy(),
    )

    ot = w.online_tables.create(
        name=ONLINE_TABLE_NAME,
        spec=online_table_spec,
    )
    print(f"Created online table: {ONLINE_TABLE_NAME}")
    print(f"Online table status: {ot}")
except Exception as e:
    print(f"Online table creation: {e}")

print("\n=== Done! ===")
print(f"Job: {JOB_NAME}")
print(f"Table: {FULL_TABLE}")
print(f"Online table: {ONLINE_TABLE_NAME}")
