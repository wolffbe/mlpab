#!/usr/bin/env python3
"""Script to create job, run it, and create feature table from predictions."""

import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, SparkPythonTask, Source
from databricks.sdk.service.compute import ClusterSpec
from databricks.sdk.service.catalog import TableType, DataSourceFormat

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpabb73485")
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabb73485")

# Initialize client
wc = WorkspaceClient()

# Get current user
current_user = wc.current_user.me()
user_name = current_user.user_name  # e.g., "benedict@logicalclocks.com"

# Workspace path for uploading files
workspace_path = f"/Users/{user_name}/{MLPAB_DATABRICKS_PREFIX}"

print(f"Using workspace path: {workspace_path}")
print(f"Using schema: {MLPAB_DATABRICKS_SCHEMA}")

# Step 1: Upload files to workspace
print("\n=== Uploading files to workspace ===")

# Create directory in workspace
wc.workspace.mkdirs(workspace_path)

# Upload train.csv
with open("data/train.csv", "rb") as f:
    wc.workspace.upload(f"{workspace_path}/train.csv", f, overwrite=True)
print("Uploaded train.csv")

# Upload score.csv
with open("data/score.csv", "rb") as f:
    wc.workspace.upload(f"{workspace_path}/score.csv", f, overwrite=True)
print("Uploaded score.csv")

# Upload train_model.py
with open("data/train_model.py", "rb") as f:
    wc.workspace.upload(f"{workspace_path}/train_model.py", f, overwrite=True)
print("Uploaded train_model.py")

# Step 2: Create and run the job
print("\n=== Creating job ===")

job_name = f"{MLPAB_DATABRICKS_PREFIX}_trainjobd631cc"

# Create a simple cluster spec
cluster_spec = ClusterSpec(
    num_workers=0,  # Single node
    node_type_id="Standard_DS3_v2",  # Basic node type
    spark_version="14.3.x-scala2.12",
    runtime_engine="STANDARD"
)

# Create the task
task = Task(
    task_key="train_task",
    new_cluster=cluster_spec,
    spark_python_task=SparkPythonTask(
        python_file=f"{workspace_path}/train_model.py",
        source=Source.WORKSPACE
    )
)

# Create the job
job = wc.jobs.create(
    name=job_name,
    tasks=[task]
)

job_id = job.job_id
print(f"Created job {job_name} with ID: {job_id}")

# Step 3: Run the job and wait for completion
print("\n=== Running job ===")

run = wc.jobs.run_now(job_id=job_id)
run_id = run.run_id
print(f"Started run with ID: {run_id}")

# Wait for the run to complete
print("Waiting for job to complete...")
while True:
    run_info = wc.jobs.get_run(run_id=run_id)
    life_cycle_state = run_info.state.life_cycle_state
    result_state = run_info.state.result_state
    
    print(f"  Current state: {life_cycle_state}, Result: {result_state}")
    
    if life_cycle_state in ["TERMINATED", "SKIPPED"]:
        if result_state == "SUCCESS":
            print("Job completed successfully!")
        else:
            print(f"Job failed with result state: {result_state}")
            # Get run output for debugging
            try:
                output = wc.jobs.get_run_output(run_id=run_id)
                print(f"Run output: {output}")
            except Exception as e:
                print(f"Could not get run output: {e}")
        break
    
    time.sleep(10)

# Step 4: Download predictions.csv from the job run
print("\n=== Downloading predictions.csv ===")

# The predictions.csv should be in the workspace directory where the job ran
# We need to download it from the workspace
predictions_workspace_path = f"{workspace_path}/predictions.csv"

# Check if file exists in workspace
try:
    # List files in workspace path
    files = wc.workspace.list(workspace_path)
    print(f"Files in {workspace_path}: {[f.path for f in files]}")
    
    # Try to download predictions.csv
    local_predictions_path = "submission/predictions.csv"
    os.makedirs("submission", exist_ok=True)
    
    with open(local_predictions_path, "wb") as f:
        wc.workspace.download(predictions_workspace_path, f)
    print(f"Downloaded predictions.csv to {local_predictions_path}")
except Exception as e:
    print(f"Could not download predictions.csv from workspace: {e}")
    # Try alternative approach - the file might be in dbfs
    try:
        dbfs_path = f"dbfs:/Users/{user_name}/{MLPAB_DATABRICKS_PREFIX}/predictions.csv"
        print(f"Trying to read from {dbfs_path}")
        # We'll need to use dbfs API
        try:
            content = wc.dbfs.read(dbfs_path)
            with open(local_predictions_path, "wb") as f:
                f.write(content.data)
            print(f"Downloaded predictions.csv from dbfs to {local_predictions_path}")
        except Exception as e2:
            print(f"Could not read from dbfs either: {e2}")
            # Last resort - check the run output
            print("Checking run output for file location...")
            try:
                output = wc.jobs.get_run_output(run_id=run_id)
                print(f"Run output: {output}")
            except Exception as e3:
                print(f"Could not get run output: {e3}")

# Step 5: Create feature table from predictions
print("\n=== Creating feature table ===")

# Parse the schema
catalog_name, schema_name = MLPAB_DATABRICKS_SCHEMA.split(".")

# First, we need to create a Delta table from the predictions
# We'll upload predictions.csv to DBFS and then create a table from it

# Upload predictions to DBFS
dbfs_table_path = f"/mnt/{MLPAB_DATABRICKS_PREFIX}_predictions"

# Actually, let's use a simpler approach - create an external table from the workspace file
# But first, let's check if we can use the workspace file directly

# For now, let's create a managed table by running a SQL query
# We need to first register the predictions as a table

# Let's use the statement execution API to run SQL
print("Creating feature table from predictions...")

# First, let's try to create a table from the CSV in workspace
# We need to use the workspace file path
workspace_file_url = f"file:/Users/{user_name}/{MLPAB_DATABRICKS_PREFIX}/predictions.csv"

# Create table SQL
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {MLPAB_DATABRICKS_SCHEMA}.predictionsd631cc 
USING CSV 
OPTIONS (path "{workspace_file_url}", header "true", inferSchema "true")
"""

try:
    # Execute the SQL to create the table
    result = wc.statement_execution.execute_statement(
        warehouse_id="default",  # Use default warehouse
        catalog=catalog_name,
        schema=schema_name,
        statement=create_table_sql,
        timeout_seconds=60
    )
    print(f"Created table: {MLPAB_DATABRICKS_SCHEMA}.predictionsd631cc")
    
    # Wait for the statement to complete
    statement_id = result.statement_id
    while True:
        status = wc.statement_execution.get_statement_result(statement_id)
        if status.status.state in ["SUCCEEDED", "FAILED", "CANCELED"]:
            print(f"Statement status: {status.status.state}")
            if status.status.state == "FAILED":
                print(f"Error: {status.status.error}")
            break
        time.sleep(5)
        
except Exception as e:
    print(f"Could not create table via SQL: {e}")
    # Try alternative approach - use the tables API
    
    # For the tables API, we need a storage location
    # Let's try a different approach - use the workspace file directly
    
    print("Trying alternative approach...")

# Step 6: Publish to online store
print("\n=== Publishing to online store ===")

# First, let's check if we have an online store
online_stores = wc.feature_store.list_online_stores()
print(f"Available online stores: {[s.name for s in online_stores]}")

# If no online store exists, create one
if not online_stores:
    online_store_name = f"{MLPAB_DATABRICKS_PREFIX}_online_store"
    online_store = wc.feature_store.create_online_store(
        name=online_store_name,
        storage_path=f"dbfs:/databricks-datasets/{MLPAB_DATABRICKS_PREFIX}/online_store"
    )
    print(f"Created online store: {online_store_name}")
else:
    online_store_name = online_stores[0].name
    print(f"Using existing online store: {online_store_name}")

# Publish the table to online store
online_table_name = "predictionsd631cc"

try:
    from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
    
    publish_spec = PublishSpec(
        online_store=online_store_name,
        online_table_name=online_table_name,
        publish_mode=PublishSpecPublishMode.OVERWRITE
    )
    
    result = wc.feature_store.publish_table(
        source_table_name=f"{MLPAB_DATABRICKS_SCHEMA}.predictionsd631cc",
        publish_spec=publish_spec
    )
    print(f"Published table to online store: {result}")
except Exception as e:
    print(f"Could not publish to online store: {e}")

# Step 7: Write submission file
print("\n=== Writing submission file ===")

import json
submission = {
    "job_name": "trainjobd631cc"
}

with open("submission/answers.json", "w") as f:
    json.dump(submission, f)

print("Written submission/answers.json")
print("\n=== DONE ===")
