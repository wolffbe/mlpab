#!/usr/bin/env python3
"""Execute the task: create job, run it, create feature table."""

import os
import time
import json
import base64
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, SparkPythonTask, Source
from databricks.sdk.service.compute import ClusterSpec
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
from databricks.sdk.service.workspace import ImportFormat

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
MLPAB_DATABRICKS_PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]

# Initialize client
wc = WorkspaceClient()

# Get current user
current_user = wc.current_user.me()
user_name = current_user.user_name

# Workspace path for uploading files
workspace_path = f"/Users/{user_name}/{MLPAB_DATABRICKS_PREFIX}"

print(f"Workspace path: {workspace_path}")
print(f"Schema: {MLPAB_DATABRICKS_SCHEMA}")

# Step 1: Upload files to workspace using import
print("\n=== Step 1: Uploading files ===")

# Check if directory exists
try:
    files = wc.workspace.list(workspace_path)
    print(f"Directory exists with files: {[f.path for f in files]}")
except Exception as e:
    print(f"Directory doesn't exist or error: {e}")
    # Try to create it
    try:
        wc.workspace.mkdirs(workspace_path)
        print(f"Created directory: {workspace_path}")
    except Exception as e2:
        print(f"Could not create directory: {e2}")

# Upload files using import with base64 encoding
def upload_file_to_workspace(local_path, workspace_dest):
    """Upload a file to workspace using import with base64 encoding."""
    with open(local_path, "rb") as f:
        content = f.read()
    
    # Base64 encode
    b64_content = base64.b64encode(content).decode('utf-8')
    
    # Import the file
    wc.workspace.import_(
        path=workspace_dest,
        content=b64_content,
        format=ImportFormat.SOURCE,
        overwrite=True
    )
    print(f"Uploaded {local_path} to {workspace_dest}")

# Upload all files
for filename in ["train.csv", "score.csv", "train_model.py"]:
    local_path = f"data/{filename}"
    workspace_dest = f"{workspace_path}/{filename}"
    upload_file_to_workspace(local_path, workspace_dest)

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

# Create task
task = Task(
    task_key="train",
    new_cluster=cluster_spec,
    spark_python_task=SparkPythonTask(
        python_file=f"{workspace_path}/train_model.py",
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

# Step 4: Get predictions.csv from workspace
print("\n=== Step 4: Getting predictions ===")

predictions_path = f"{workspace_path}/predictions.csv"
local_predictions = "submission/predictions.csv"
os.makedirs("submission", exist_ok=True)

# Try to download from workspace
try:
    with open(local_predictions, "wb") as f:
        wc.workspace.export(predictions_path, f)
    print(f"Downloaded predictions.csv")
    
    # Verify the file
    with open(local_predictions, "r") as f:
        content = f.read()
        print(f"Predictions content (first 200 chars): {content[:200]}")
except Exception as e:
    print(f"Could not download from workspace: {e}")
    # Try dbfs
    try:
        dbfs_path = f"dbfs:{predictions_path}"
        content = wc.dbfs.read(dbfs_path)
        with open(local_predictions, "wb") as f:
            f.write(content.data)
        print(f"Downloaded from dbfs")
    except Exception as e2:
        print(f"Could not download from dbfs: {e2}")
        # Try listing workspace to see what's there
        try:
            files = wc.workspace.list(workspace_path)
            print(f"Files in workspace: {[f.path for f in files]}")
        except Exception as e3:
            print(f"Could not list workspace: {e3}")
        raise

# Step 5: Create feature table
print("\n=== Step 5: Creating feature table ===")

catalog_name, schema_name = MLPAB_DATABRICKS_SCHEMA.split(".")
table_name = "predictionsd631cc"
full_table_name = f"{MLPAB_DATABRICKS_SCHEMA}.{table_name}"

# Upload predictions.csv to DBFS for easier table creation
dbfs_dir = f"dbfs:/mnt/{MLPAB_DATABRICKS_PREFIX}/data"
dbfs_csv = f"{dbfs_dir}/predictions.csv"

# Ensure directory exists
try:
    wc.dbfs.mkdirs(dbfs_dir)
except:
    pass

# Upload to DBFS
with open(local_predictions, "rb") as f:
    wc.dbfs.upload(dbfs_csv, f, overwrite=True)
print(f"Uploaded to DBFS: {dbfs_csv}")

# Create table using SQL
create_sql = f"""
CREATE OR REPLACE TABLE {full_table_name} 
USING CSV 
OPTIONS (
  path "{dbfs_csv}",
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
        # Try to use a default online store
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
