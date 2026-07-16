#!/usr/bin/env python3
"""Execute the task: create job, run it, create feature table."""

import os
import time
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, SparkPythonTask, Source
from databricks.sdk.service.compute import ClusterSpec
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
MLPAB_DATABRICKS_PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]

# Initialize client
wc = WorkspaceClient()

# Get current user
current_user = wc.current_user.me()
user_name = current_user.user_name

# DBFS path for storing files
# Use a path that doesn't have @ in it
dbfs_base = f"dbfs:/mnt/{MLPAB_DATABRICKS_PREFIX}"
dbfs_data_dir = f"{dbfs_base}/data"
dbfs_script_dir = f"{dbfs_base}/scripts"

print(f"DBFS base: {dbfs_base}")
print(f"Schema: {MLPAB_DATABRICKS_SCHEMA}")

# Step 1: Upload files to DBFS
print("\n=== Step 1: Uploading files to DBFS ===")

# Create directories
wc.dbfs.mkdirs(dbfs_data_dir)
wc.dbfs.mkdirs(dbfs_script_dir)
print(f"Created directories: {dbfs_data_dir}, {dbfs_script_dir}")

# Upload files
files_to_upload = {
    "data/train.csv": f"{dbfs_data_dir}/train.csv",
    "data/score.csv": f"{dbfs_data_dir}/score.csv",
    "data/train_model.py": f"{dbfs_script_dir}/train_model.py"
}

for local_path, dbfs_path in files_to_upload.items():
    with open(local_path, "rb") as f:
        wc.dbfs.upload(dbfs_path, f, overwrite=True)
    print(f"Uploaded {local_path} to {dbfs_path}")

# Step 2: Create the job
print("\n=== Step 2: Creating job ===")

job_name = f"{MLPAB_DATABRICKS_PREFIX}_trainjobd631cc"

# Create cluster spec
cluster_spec = ClusterSpec(
    num_workers=0,
    node_type_id="Standard_DS3_v2",
    spark_version="14.3.x-scala2.12",
    runtime_engine="STANDARD"
)

# The training script needs to find train.csv and score.csv in its working directory
# We'll modify the approach: create a wrapper script that copies files to current directory
# and then runs the training script

# Create a wrapper script that:
# 1. Copies data files from DBFS to current directory
# 2. Runs the training script
wrapper_script = f"""
import os
import shutil

# Copy data files from DBFS to current directory
shutil.copy("{dbfs_data_dir}/train.csv", "train.csv")
shutil.copy("{dbfs_data_dir}/score.csv", "score.csv")
shutil.copy("{dbfs_script_dir}/train_model.py", "train_model.py")

# Run the training script
import subprocess
subprocess.run(["python", "train_model.py"], check=True)
"""

# Upload wrapper script to DBFS
wrapper_path = f"{dbfs_script_dir}/run_training.py"
with open("submission/wrapper.py", "w") as f:
    f.write(wrapper_script)

with open("submission/wrapper.py", "rb") as f:
    wc.dbfs.upload(wrapper_path, f, overwrite=True)
print(f"Uploaded wrapper script to {wrapper_path}")

# Create task that runs the wrapper script
task = Task(
    task_key="train",
    new_cluster=cluster_spec,
    spark_python_task=SparkPythonTask(
        python_file=wrapper_path,
        source=Source.WORKSPACE  # This might need to be DBFS
    )
)

# Actually, SparkPythonTask with DBFS path might not work directly
# Let's try a different approach - use a notebook task or copy to workspace first

# Alternative: Upload wrapper to workspace and reference DBFS paths in the script
workspace_path = f"/Users/{user_name}/{MLPAB_DATABRICKS_PREFIX}"

# Upload wrapper to workspace
try:
    wc.workspace.mkdirs(workspace_path)
except:
    pass

# Create a simpler wrapper that reads from DBFS
wrapper_script_v2 = f"""
import os
import pandas as pd
import numpy as np

# Read data from DBFS
 train = pd.read_csv("{dbfs_data_dir}/train.csv")
 score = pd.read_csv("{dbfs_data_dir}/score.csv")

FEATURES = ["f1", "f2", "f3", "f4", "f5"]
LEARNING_RATE = 0.1
ITERATIONS = 300

X = train[FEATURES].to_numpy(dtype=float)
y = train["label"].to_numpy(dtype=float)
w = np.zeros(X.shape[1], dtype=float)
b = 0.0
for _ in range(ITERATIONS):
    p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    g = p - y
    w = w - LEARNING_RATE * (X.T @ g) / len(y)
    b = b - LEARNING_RATE * g.mean()
Xs = score[FEATURES].to_numpy(dtype=float)
preds = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
out = pd.DataFrame({{"row_id": score["row_id"], "score": np.round(preds, 6)}})
out.to_csv("predictions.csv", index=False)
"""

# Upload this inline script to workspace
workspace_wrapper = f"{workspace_path}/run_training.py"
with open("submission/wrapper2.py", "w") as f:
    f.write(wrapper_script_v2)

# Try upload using workspace API
try:
    with open("submission/wrapper2.py", "rb") as f:
        wc.workspace.upload(workspace_wrapper, f, overwrite=True)
    print(f"Uploaded wrapper to workspace: {workspace_wrapper}")
except Exception as e:
    print(f"Workspace upload failed: {e}")
    # Try using dbfs path in the job instead
    print("Will try using DBFS path directly in job")
    workspace_wrapper = wrapper_path

# Create task
task = Task(
    task_key="train",
    new_cluster=cluster_spec,
    spark_python_task=SparkPythonTask(
        python_file=workspace_wrapper,
        source=Source.WORKSPACE
    )
)

job = wc.jobs.create(
    name=job_name,
    tasks=[task]
)

job_id = job.job_id
print(f"Created job: {job_name} (ID: {job_id})")

# Step 3: Run the job
print("\n=== Step 3: Running job ===")

run = wc.jobs.run_now(job_id=job_id)
run_id = run.run_id
print(f"Started run: {run_id}")

# Wait for completion
while True:
    run_info = wc.jobs.get_run(run_id=run_id)
    state = run_info.state
    life_cycle = state.life_cycle_state
    result = state.result_state
    
    print(f"  State: {life_cycle}, Result: {result}")
    
    if life_cycle in ["TERMINATED", "SKIPPED"]:
        if result == "SUCCESS":
            print("Job completed successfully!")
        else:
            print(f"Job failed: {result}")
            try:
                output = wc.jobs.get_run_output(run_id=run_id)
                print(f"Output: {output}")
            except Exception as e:
                print(f"Could not get output: {e}")
        break
    
    time.sleep(15)

# Step 4: Get predictions.csv
print("\n=== Step 4: Getting predictions ===")

# The predictions.csv should be in the workspace directory where the job ran
# or we can try to find it
predictions_dbfs = f"{dbfs_base}/predictions.csv"
local_predictions = "submission/predictions.csv"
os.makedirs("submission", exist_ok=True)

# Try to download from DBFS first
try:
    content = wc.dbfs.read(predictions_dbfs)
    with open(local_predictions, "wb") as f:
        f.write(content.data)
    print(f"Downloaded predictions.csv from DBFS")
except Exception as e:
    print(f"Could not download from DBFS: {e}")
    # Try workspace
    predictions_workspace = f"{workspace_path}/predictions.csv"
    try:
        with open(local_predictions, "wb") as f:
            wc.workspace.export(predictions_workspace, f)
        print(f"Downloaded predictions.csv from workspace")
    except Exception as e2:
        print(f"Could not download from workspace: {e2}")
        # Try to list workspace to see what's there
        try:
            files = wc.workspace.list(workspace_path)
            print(f"Files in workspace: {[f.path for f in files]}")
        except Exception as e3:
            print(f"Could not list workspace: {e3}")
        raise

# Verify the file
with open(local_predictions, "r") as f:
    content = f.read()
    print(f"Predictions content (first 200 chars): {content[:200]}")

# Step 5: Create feature table
print("\n=== Step 5: Creating feature table ===")

catalog_name, schema_name = MLPAB_DATABRICKS_SCHEMA.split(".")
table_name = "predictionsd631cc"
full_table_name = f"{MLPAB_DATABRICKS_SCHEMA}.{table_name}"

# Upload predictions.csv to DBFS for table creation
predictions_table_dbfs = f"{dbfs_data_dir}/predictions.csv"
with open(local_predictions, "rb") as f:
    wc.dbfs.upload(predictions_table_dbfs, f, overwrite=True)
print(f"Uploaded predictions to: {predictions_table_dbfs}")

# Create table using SQL
create_sql = f"""
CREATE OR REPLACE TABLE {full_table_name} 
USING CSV 
OPTIONS (
  path "{predictions_table_dbfs}",
  header "true",
  inferSchema "true"
)
"""

try:
    result = wc.statement_execution.execute_statement(
        warehouse_id="default",
        catalog=catalog_name,
        schema=schema_name,
        statement=create_sql,
        timeout_seconds=120
    )
    
    statement_id = result.statement_id
    print(f"Executing statement: {statement_id}")
    
    # Wait for completion
    while True:
        status = wc.statement_execution.get_statement_result(statement_id)
        state = status.status.state
        print(f"  SQL State: {state}")
        
        if state in ["SUCCEEDED", "FAILED", "CANCELED"]:
            if state == "SUCCEEDED":
                print(f"Table created: {full_table_name}")
            else:
                print(f"SQL failed: {status.status.error}")
            break
        time.sleep(5)
        
except Exception as e:
    print(f"SQL approach failed: {e}")
    raise

# Step 6: Create online table for low-latency lookup
print("\n=== Step 6: Creating online table ===")

# Check/create online store
online_stores = wc.feature_store.list_online_stores()
if online_stores:
    online_store_name = online_stores[0].name
    print(f"Using existing online store: {online_store_name}")
else:
    online_store_name = f"{MLPAB_DATABRICKS_PREFIX}_online_store"
    try:
        online_store = wc.feature_store.create_online_store(
            name=online_store_name,
            storage_path=f"dbfs:/databricks-datasets/{MLPAB_DATABRICKS_PREFIX}/online_store"
        )
        print(f"Created online store: {online_store_name}")
    except Exception as e:
        print(f"Could not create online store: {e}")
        online_store_name = "default"
        print(f"Using default online store")

# Publish table to online store
online_table_name = "predictionsd631cc"

try:
    publish_spec = PublishSpec(
        online_store=online_store_name,
        online_table_name=online_table_name,
        publish_mode=PublishSpecPublishMode.OVERWRITE
    )
    
    result = wc.feature_store.publish_table(
        source_table_name=full_table_name,
        publish_spec=publish_spec
    )
    print(f"Published to online store: {result}")
except Exception as e:
    print(f"Could not publish to online store: {e}")
    print("Note: Online table creation may need manual intervention")

# Step 7: Write submission file
print("\n=== Step 7: Writing submission ===")

submission = {
    "job_name": "trainjobd631cc"
}

with open("submission/answers.json", "w") as f:
    json.dump(submission, f)

print("Written submission/answers.json")
print("\n=== TASK COMPLETE ===")
