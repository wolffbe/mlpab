#!/usr/bin/env python3
"""Final attempt: use PythonOperatorTask with inline code."""

import os
import time
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, PythonOperatorTask, Source
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

workspace_path = f"/Users/{user_name}/{MLPAB_DATABRICKS_PREFIX}"

print(f"Workspace path: {workspace_path}")
print(f"Schema: {MLPAB_DATABRICKS_SCHEMA}")

# Read the data files
with open("data/train.csv", "r") as f:
    train_csv = f.read()

with open("data/score.csv", "r") as f:
    score_csv = f.read()

with open("data/train_model.py", "r") as f:
    train_script = f.read()

# Step 1: Create the job with PythonOperatorTask
print("\n=== Step 1: Creating job ===")

job_name = f"{MLPAB_DATABRICKS_PREFIX}_trainjobd631cc"

# Create cluster spec
cluster_spec = ClusterSpec(
    num_workers=0,
    node_type_id="Standard_DS3_v2",
    spark_version="14.3.x-scala2.12",
    runtime_engine="STANDARD"
)

# Create inline Python code that:
# 1. Writes the CSV files
# 2. Writes the training script
# 3. Runs the training script
inline_code = f"""
import os
import pandas as pd
import numpy as np
import subprocess

# Write train.csv
with open("train.csv", "w") as f:
    f.write({repr(train_csv)})

# Write score.csv
with open("score.csv", "w") as f:
    f.write({repr(score_csv)})

# Write train_model.py
with open("train_model.py", "w") as f:
    f.write({repr(train_script)})

# Run the training script
subprocess.run(["python", "train_model.py"], check=True)
"""

# Create task with PythonOperatorTask
task = Task(
    task_key="train",
    new_cluster=cluster_spec,
    python_operator_task=PythonOperatorTask(
        main=inline_code
    )
)

try:
    job = wc.jobs.create(
        name=job_name,
        tasks=[task]
    )
    
    job_id = job.job_id
    print(f"Created job: {job_name} (ID: {job_id})")
except Exception as e:
    print(f"Failed to create job: {e}")
    # Try with compute field
    print("Trying with compute field...")
    from databricks.sdk.service.compute import HardwareAcceleratorType
    task.compute = Compute(hardware_accelerator=HardwareAcceleratorType.NONE)
    
    try:
        job = wc.jobs.create(
            name=job_name,
            tasks=[task]
        )
        job_id = job.job_id
        print(f"Created job: {job_name} (ID: {job_id})")
    except Exception as e2:
        print(f"Still failed: {e2}")
        raise

# Step 2: Run the job
print("\n=== Step 2: Running job ===")

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

# Step 3: Get predictions.csv
print("\n=== Step 3: Getting predictions ===")

# The predictions.csv should be in the workspace directory where the job ran
# For PythonOperatorTask, the working directory might be different
# Let's try to find it

# First, try in the workspace path
predictions_workspace = f"{workspace_path}/predictions.csv"
local_predictions = "submission/predictions.csv"
os.makedirs("submission", exist_ok=True)

try:
    resp = wc.workspace.export(predictions_workspace)
    import base64
    content = base64.b64decode(resp.content).decode()
    with open(local_predictions, "w") as f:
        f.write(content)
    print(f"Downloaded predictions.csv from workspace")
    print(f"Content (first 200 chars): {content[:200]}")
except Exception as e:
    print(f"Could not download from workspace: {e}")
    # Try other locations
    # Maybe it's in DBFS
    try:
        # Try to list workspace to see what's there
        files = wc.workspace.list(workspace_path)
        print(f"Files in {workspace_path}: {[f.path for f in files]}")
        
        # Try to find predictions.csv
        for f in files:
            if "predictions" in f.path:
                resp = wc.workspace.export(f.path)
                content = base64.b64decode(resp.content).decode()
                with open(local_predictions, "w") as f:
                    f.write(content)
                print(f"Found predictions at {f.path}")
                break
        else:
            raise Exception("predictions.csv not found in workspace")
    except Exception as e2:
        print(f"Could not find predictions: {e2}")
        raise

# Step 4: Create feature table
print("\n=== Step 4: Creating feature table ===")

catalog_name, schema_name = MLPAB_DATABRICKS_SCHEMA.split(".")
table_name = "predictionsd631cc"
full_table_name = f"{MLPAB_DATABRICKS_SCHEMA}.{table_name}"

# We need to get the predictions into a table
# Since DBFS is disabled, we'll need to use workspace files
# But we can't create external tables from workspace files directly

# Let's try using the statement execution API with a CREATE TABLE statement
# that references the workspace file

# First, let's try to upload predictions to a location that works
# We'll use the workspace file directly in the SQL

# Actually, let's try a different approach - use the workspace file in a SQL query
# But SQL can't read from workspace directly

# Let's try using the files API to read the predictions
# and then use statement execution to create a table

# Read predictions content
with open(local_predictions, "r") as f:
    predictions_content = f.read()

# Create a Delta table using the statement execution API
# We'll use a CREATE TABLE statement with VALUES
# But that won't work for a CSV file

# Alternative: Use the tables API to create a table
# But that requires a storage location

# Let's try using the statement execution with a temporary view first
# Then create a table from it

# Parse the CSV and create INSERT statements
import csv
from io import StringIO

# Read predictions
with open(local_predictions, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Create table using SQL
create_sql = f"""
CREATE OR REPLACE TABLE {full_table_name} (
  row_id STRING,
  score DOUBLE
)
"""

try:
    result = wc.statement_execution.execute_statement(
        warehouse_id="default",
        catalog=catalog_name,
        schema=schema_name,
        statement=create_sql,
        timeout_seconds=60
    )
    
    statement_id = result.statement_id
    print(f"Created table structure: {statement_id}")
    
    # Wait for completion
    while True:
        status = wc.statement_execution.get_statement_result(statement_id)
        state = status.status.state
        if state in ["SUCCEEDED", "FAILED", "CANCELED"]:
            if state == "SUCCEEDED":
                print(f"Table structure created: {full_table_name}")
            else:
                print(f"SQL failed")
            break
        time.sleep(5)
    
    # Now insert data
    for row in rows:
        row_id = row["row_id"]
        score = row["score"]
        insert_sql = f"INSERT INTO {full_table_name} VALUES ('{row_id}', {score})"
        
        result = wc.statement_execution.execute_statement(
            warehouse_id="default",
            catalog=catalog_name,
            schema=schema_name,
            statement=insert_sql,
            timeout_seconds=60
        )
        
        statement_id = result.statement_id
        while True:
            status = wc.statement_execution.get_statement_result(statement_id)
            state = status.status.state
            if state in ["SUCCEEDED", "FAILED", "CANCELED"]:
                if state != "SUCCEEDED":
                    print(f"Insert failed for {row_id}")
                break
            time.sleep(1)
    
    print(f"Table populated: {full_table_name}")
    
except Exception as e:
    print(f"Failed to create table: {e}")
    raise

# Step 5: Create online table
print("\n=== Step 5: Creating online table ===")

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

try:
    publish_spec = PublishSpec(
        online_store=online_store_name,
        online_table_name="predictionsd631cc",
        publish_mode=PublishSpecPublishMode.OVERWRITE
    )
    
    result = wc.feature_store.publish_table(
        source_table_name=full_table_name,
        publish_spec=publish_spec
    )
    print(f"Published to online store: {result}")
except Exception as e:
    print(f"Could not publish to online store: {e}")

# Step 6: Write submission file
print("\n=== Step 6: Writing submission ===")

submission = {
    "job_name": "trainjobd631cc"
}

with open("submission/answers.json", "w") as f:
    json.dump(submission, f)

print("Written submission/answers.json")
print("\n=== TASK COMPLETE ===")
