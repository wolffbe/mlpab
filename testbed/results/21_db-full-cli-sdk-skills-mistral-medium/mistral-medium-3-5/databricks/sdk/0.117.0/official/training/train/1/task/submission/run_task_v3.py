#!/usr/bin/env python3
"""Script to run the training job and create feature table on Databricks platform."""

import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, jobs, compute
from databricks.sdk.service.workspace import ImportFormat

def main():
    # Initialize workspace client
    w = WorkspaceClient()
    
    # Environment variables
    prefix = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabdc2d18")
    schema = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpabdc2d18")
    
    print(f"Prefix: {prefix}")
    print(f"Schema: {schema}")
    
    # Get current user
    user = w.current_user.me().user_name
    workspace_path = f"/Users/{user}/{prefix}/"
    print(f"Workspace path: {workspace_path}")
    
    # Step 1: Upload data files and training script to workspace
    print("\n=== Uploading files ===")
    # Create directory if it doesn't exist
    w.workspace.mkdirs(workspace_path)
    print(f"Created directory: {workspace_path}")
    
    with open("data/train.csv", "rb") as f:
        w.workspace.upload(f"{workspace_path}train.csv", f.read(), format=ImportFormat.RAW, overwrite=True)
    print("Uploaded train.csv")
    
    with open("data/score.csv", "rb") as f:
        w.workspace.upload(f"{workspace_path}score.csv", f.read(), format=ImportFormat.RAW, overwrite=True)
    print("Uploaded score.csv")
    
    with open("data/train_model.py", "rb") as f:
        w.workspace.upload(f"{workspace_path}train_model.py", f.read(), format=ImportFormat.RAW, overwrite=True)
    print("Uploaded train_model.py")
    
    # Step 2: Create a job to run the training script
    print("\n=== Creating job ===")
    job_name = f"{prefix}_trainjobd631cc"
    
    # Check if job already exists and delete it
    existing_jobs = list(w.jobs.list(name=job_name))
    if existing_jobs:
        job_id = existing_jobs[0].job_id
        print(f"Deleting existing job {job_id}")
        w.jobs.delete(job_id)
    
    # Create new job with WORKSPACE source and serverless compute
    # Create environment for serverless task
    environment_key = f"{prefix}_env"
    job_environment = jobs.JobEnvironment(
        environment_key=environment_key,
        spec=compute.Environment(
            base_environment="workspace-base-environments/databricks_ml",
        ),
    )
    
    spark_python_task = jobs.SparkPythonTask(
        python_file=f"{workspace_path}train_model.py",
        source=jobs.Source.WORKSPACE,
    )
    
    task = jobs.Task(
        task_key="train",
        compute=jobs.Compute(),
        spark_python_task=spark_python_task,
        environment_key=environment_key,
    )
    
    job = w.jobs.create(
        name=job_name,
        tasks=[task],
        environments=[job_environment],
    )
    
    job_id = job.job_id
    print(f"Created job {job_name} with ID: {job_id}")
    
    # Step 3: Run the job
    print("\n=== Running job ===")
    run = w.jobs.run_now(job_id)
    run_id = run.run_id
    print(f"Started job run with ID: {run_id}")
    
    # Wait for job to complete
    print("Waiting for job to complete...")
    while True:
        run_info = w.jobs.get_run(run_id)
        state = run_info.state
        if state.life_cycle_state in ["TERMINATED", "INTERNAL_ERROR"]:
            print(f"Job run {run_id} completed with state: {state.life_cycle_state}, result: {state.result_state}")
            break
        time.sleep(10)
    
    # Check if job succeeded
    if state.result_state != "SUCCESS":
        print(f"Job failed with result state: {state.result_state}")
        print(f"State message: {state.state_message}")
        raise Exception(f"Job failed: {state.result_state}")
    
    print("Job completed successfully")
    
    # Step 4: Get the predictions.csv from the job run
    # The predictions.csv should be in the workspace at the working directory
    print("\n=== Retrieving predictions ===")
    
    # Try to download predictions.csv from workspace
    try:
        predictions_content = w.workspace.download(f"{workspace_path}predictions.csv")
        print("Found predictions.csv in workspace")
        
        # Save locally for inspection
        os.makedirs("submission", exist_ok=True)
        with open("submission/predictions.csv", "wb") as f:
            f.write(predictions_content)
        print("Saved predictions.csv locally")
    except Exception as e:
        print(f"Could not find predictions.csv in workspace: {e}")
        # Try DBFS path
        dbfs_workspace_path = workspace_path.replace("/Users/", "dbfs:/user/")
        try:
            predictions_content = w.dbfs.read(dbfs_workspace_path + "predictions.csv")
            print("Found predictions.csv in DBFS")
            with open("submission/predictions.csv", "wb") as f:
                f.write(predictions_content)
        except Exception as e2:
            print(f"Could not find predictions.csv in DBFS either: {e2}")
            raise Exception("Could not locate predictions.csv output")
    
    # Step 5: Create Delta table in Unity Catalog
    print("\n=== Creating Delta table ===")
    
    catalog_name = schema.split(".")[0]  # "workspace"
    schema_name = schema.split(".")[1]    # "mlpabdc2d18"
    
    # Check if schema exists, create if not
    try:
        w.schemas.get(catalog_name, schema_name)
        print(f"Schema {schema} exists")
    except Exception as e:
        print(f"Creating schema {schema}: {e}")
        w.schemas.create(catalog_name, schema_name)
        print(f"Created schema {schema}")
    
    # Upload predictions to DBFS
    dbfs_predictions_path = f"dbfs:/tmp/{prefix}_predictions.csv"
    w.dbfs.upload(dbfs_predictions_path, predictions_content, overwrite=True)
    print(f"Uploaded predictions to DBFS: {dbfs_predictions_path}")
    
    # Create Delta table from CSV
    table_name = f"{schema}.predictionsd631cc"
    
    # Create the Delta table
    sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} 
    (row_id STRING, score DOUBLE)
    USING DELTA
    COMMENT 'Predictions from trainjobd631cc'
    """
    
    w.statement_execution.execute_statement(
        warehouse_id="default",
        catalog=catalog_name,
        schema=schema_name,
        statement=sql,
    )
    print(f"Created Delta table: {table_name}")
    
    # Import data from CSV
    sql = f"""
    COPY INTO {table_name} 
    FROM (SELECT * FROM csv.`{dbfs_predictions_path.replace('dbfs:', '/')}`)
    FILEFORMAT OPTIONS (header "true", inferSchema "true")
    """
    
    w.statement_execution.execute_statement(
        warehouse_id="default",
        catalog=catalog_name,
        schema=schema_name,
        statement=sql,
    )
    print(f"Imported data into Delta table")
    
    # Step 6: Create online table for low-latency lookup
    print("\n=== Creating online table ===")
    
    online_table_name = f"{prefix}_predictionsd631cc_online"
    
    # Create online table
    online_table = catalog.OnlineTable(
        name=online_table_name,
        spec=catalog.OnlineTableSpec(
            source_table_full_name=table_name,
            primary_key_columns=["row_id"],
        ),
    )
    
    result = w.online_tables.create_and_wait(online_table)
    print(f"Created online table: {online_table_name}")
    print(f"Online table status: {result.status}")
    
    # Step 7: Write submission/answers.json
    print("\n=== Writing answers.json ===")
    os.makedirs("submission", exist_ok=True)
    answers = {"job_name": "trainjobd631cc"}
    with open("submission/answers.json", "w") as f:
        json.dump(answers, f)
    print(f"Written submission/answers.json: {answers}")
    
    print("\n=== Task completed successfully! ===")

if __name__ == "__main__":
    main()
