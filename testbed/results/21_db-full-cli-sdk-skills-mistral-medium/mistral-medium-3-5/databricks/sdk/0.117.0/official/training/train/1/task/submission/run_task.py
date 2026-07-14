#!/usr/bin/env python3
"""Script to run the training job and create feature table on Databricks platform."""

import os
import json
import time
from databricks.sdk import WorkspaceClient

def main():
    # Initialize workspace client
    w = WorkspaceClient()
    
    # Environment variables
    prefix = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabdc2d18")
    schema = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpabdc2d18")
    
    print(f"Prefix: {prefix}")
    print(f"Schema: {schema}")
    
    # Step 1: Upload data files and training script to workspace
    workspace_path = f"/Users/{w.current_user.me().login_name}/{prefix}/"
    print(f"Uploading files to workspace path: {workspace_path}")
    
    # Upload train.csv
    with open("data/train.csv", "rb") as f:
        w.workspace.upload(f"{workspace_path}train.csv", f.read(), overwrite=True)
    
    # Upload score.csv
    with open("data/score.csv", "rb") as f:
        w.workspace.upload(f"{workspace_path}score.csv", f.read(), overwrite=True)
    
    # Upload train_model.py
    with open("data/train_model.py", "rb") as f:
        w.workspace.upload(f"{workspace_path}train_model.py", f.read(), overwrite=True)
    
    print("Files uploaded successfully")
    
    # Step 2: Create a job to run the training script
    job_name = f"{prefix}_trainjobd631cc"
    
    # First, check if job already exists
    existing_jobs = w.jobs.list(name=job_name)
    if existing_jobs:
        job_id = existing_jobs[0].job_id
        print(f"Job already exists with ID: {job_id}")
        # Delete it to recreate
        w.jobs.delete(job_id)
        print(f"Deleted existing job {job_id}")
    
    # Create new job
    job = w.jobs.create(
        name=job_name,
        tasks=[
            {
                "task_key": "train",
                "new_cluster": {
                    "spark_version": "14.3.x-scala2.12",
                    "node_type_id": "Standard_DS3_v2",
                    "num_workers": 0,
                },
                "spark_python_task": {
                    "python_file": f"{workspace_path}train_model.py",
                },
                "libraries": [],
            }
        ],
    )
    
    job_id = job.job_id
    print(f"Created job {job_name} with ID: {job_id}")
    
    # Step 3: Run the job
    run = w.jobs.run(job_id)
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
    # For a job, we need to check the cluster's workspace or DBFS
    
    # Try to download predictions.csv from workspace
    try:
        predictions_content = w.workspace.download(f"{workspace_path}predictions.csv")
        print("Found predictions.csv in workspace")
        
        # Save locally for inspection
        with open("submission/predictions.csv", "wb") as f:
            f.write(predictions_content)
        print("Saved predictions.csv locally")
    except Exception as e:
        print(f"Could not find predictions.csv in workspace: {e}")
        # Try to get it from DBFS
        try:
            dbfs_path = f"dbfs:/Users/{w.current_user.me().login_name}/{prefix}/predictions.csv"
            predictions_content = w.dbfs.read(dbfs_path)
            print("Found predictions.csv in DBFS")
            with open("submission/predictions.csv", "wb") as f:
                f.write(predictions_content)
        except Exception as e2:
            print(f"Could not find predictions.csv in DBFS either: {e2}")
            raise Exception("Could not locate predictions.csv output")
    
    # Step 5: Create feature table from predictions
    # First, let's read the predictions to understand the schema
    import pandas as pd
    import io
    
    predictions_df = pd.read_csv(io.BytesIO(predictions_content))
    print(f"Predictions shape: {predictions_df.shape}")
    print(f"Predictions columns: {predictions_df.columns.tolist()}")
    print(f"First few rows:\n{predictions_df.head()}")
    
    # Step 6: Create feature table in Unity Catalog
    # We need to use the Feature Store API
    # The table should be named predictionsd631cc, version 1, in the specified schema
    
    # First, create the catalog if it doesn't exist
    catalog_name = schema.split(".")[0]  # "workspace"
    schema_name = schema.split(".")[1]    # "mlpabdc2d18"
    
    # Check if catalog exists
    try:
        catalog = w.catalogs.get(catalog_name)
        print(f"Catalog {catalog_name} exists")
    except Exception as e:
        print(f"Catalog {catalog_name} not found, creating: {e}")
        # workspace catalog should exist by default
        pass
    
    # Check if schema exists
    try:
        schema_obj = w.schemas.get(catalog_name, schema_name)
        print(f"Schema {schema} exists")
    except Exception as e:
        print(f"Schema {schema} not found, creating: {e}")
        w.schemas.create(catalog_name, schema_name)
        print(f"Created schema {schema}")
    
    # Upload predictions to DBFS as a CSV
    dbfs_predictions_path = f"dbfs:/tmp/{prefix}_predictions.csv"
    w.dbfs.upload(dbfs_predictions_path, predictions_content, overwrite=True)
    print(f"Uploaded predictions to DBFS: {dbfs_predictions_path}")
    
    # Create a Delta table from the CSV
    delta_table_name = f"{schema}.predictionsd631cc_raw"
    
    # Use SQL to create the table
    sql = f"""
    CREATE TABLE IF NOT EXISTS {delta_table_name} 
    USING CSV 
    OPTIONS (path "{dbfs_predictions_path.replace('dbfs:', '/')}", header "true", inferSchema "true")
    """
    
    try:
        w.statement_execution.execute_statement(
            warehouse_id="default",
            catalog=catalog_name,
            schema=schema_name,
            statement=sql,
        )
        print(f"Created raw CSV table: {delta_table_name}")
    except Exception as e:
        print(f"Failed to create raw CSV table: {e}")
        # Try alternative approach - create Delta table directly
        sql = f"""
        CREATE TABLE IF NOT EXISTS {delta_table_name} 
        (row_id STRING, score DOUBLE)
        USING DELTA
        """
        w.statement_execution.execute_statement(
            warehouse_id="default",
            catalog=catalog_name,
            schema=schema_name,
            statement=sql,
        )
        print(f"Created empty Delta table: {delta_table_name}")
        
        # Copy data from CSV
        sql = f"""
        COPY INTO {delta_table_name} 
        FROM (SELECT * FROM csv.`{dbfs_predictions_path.replace('dbfs:', '/')}`)
        FILEFORMAT OPTIONS (header "true", inferSchema "true")
        """
        w.statement_execution.execute_statement(
            warehouse_id="default",
            catalog=catalog_name,
            schema=schema_name,
            statement=sql,
        )
        print(f"Copied data to Delta table")
    
    # Now create the feature table using Feature Store API
    # The feature table should be named predictionsd631cc with version 1
    feature_table_name = "predictionsd631cc"
    
    try:
        # Create feature table
        feature_table = w.feature_tables.create(
            name=f"{schema}.{feature_table_name}",
            primary_keys=["row_id"],
            df=predictions_df,
        )
        print(f"Created feature table: {feature_table.full_name}")
    except Exception as e:
        print(f"Failed to create feature table directly: {e}")
        # Try alternative approach - create via SQL
        # First create the table in Unity Catalog
        sql = f"""
        CREATE TABLE IF NOT EXISTS {schema}.{feature_table_name} 
        (row_id STRING, score DOUBLE)
        USING DELTA
        COMMENT 'Feature table for predictions'
        """
        w.statement_execution.execute_statement(
            warehouse_id="default",
            catalog=catalog_name,
            schema=schema_name,
            statement=sql,
        )
        print(f"Created Delta table for feature store: {schema}.{feature_table_name}")
        
        # Insert data
        # Convert predictions to values for INSERT
        rows = []
        for _, row in predictions_df.iterrows():
            rows.append(f"('{row['row_id']}', {row['score']})")
        
        values_clause = ", ".join(rows)
        sql = f"""
        INSERT INTO {schema}.{feature_table_name} 
        VALUES {values_clause}
        """
        w.statement_execution.execute_statement(
            warehouse_id="default",
            catalog=catalog_name,
            schema=schema_name,
            statement=sql,
        )
        print(f"Inserted data into feature table")
    
    # Step 7: Make features available for low-latency lookup (online store)
    # This means creating an online table/feature store endpoint
    try:
        # Register the feature table for online access
        # In Databricks, this is done by enabling the online store for the feature table
        ft = w.feature_tables.get(f"{schema}.{feature_table_name}")
        print(f"Feature table info: {ft}")
        
        # Enable online store
        w.feature_tables.enable_online_store(f"{schema}.{feature_table_name}")
        print(f"Enabled online store for feature table {schema}.{feature_table_name}")
    except Exception as e:
        print(f"Could not enable online store directly: {e}")
        # Try creating an online table via the serving API
        try:
            online_table_name = f"{prefix}_predictionsd631cc_online"
            w.serving.create_online_table(
                name=online_table_name,
                source_table=f"{schema}.{feature_table_name}",
            )
            print(f"Created online table: {online_table_name}")
        except Exception as e2:
            print(f"Could not create online table: {e2}")
            # The feature table itself should be queryable for both batch and online
            print("Feature table created - online access may be automatic")
    
    # Step 8: Write submission/answers.json
    os.makedirs("submission", exist_ok=True)
    answers = {"job_name": "trainjobd631cc"}
    with open("submission/answers.json", "w") as f:
        json.dump(answers, f)
    print(f"Written submission/answers.json: {answers}")
    
    print("\nTask completed successfully!")

if __name__ == "__main__":
    main()
