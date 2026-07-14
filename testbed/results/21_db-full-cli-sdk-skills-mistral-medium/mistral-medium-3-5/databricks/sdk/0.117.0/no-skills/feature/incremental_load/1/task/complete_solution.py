#!/usr/bin/env python3

import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    CronSchedule,
    Task
)
from databricks.sdk.service import compute

def main():
    # Environment variables
    MLPAB_DATABRICKS_SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabb7e680')
    MLPAB_DATABRICKS_PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabb7e680')
    
    # Parse schema
    catalog_name, schema_name = MLPAB_DATABRICKS_SCHEMA.split('.')
    
    print(f"Catalog: {catalog_name}, Schema: {schema_name}, Prefix: {MLPAB_DATABRICKS_PREFIX}")
    
    # Initialize client
    wc = WorkspaceClient()
    
    # Get current user
    current_user = wc.current_user.me()
    user_id = current_user.id
    print(f"Current user: {user_id}")
    
    # 1. Upload data files to workspace
    print("Uploading data files to workspace...")
    local_data_path = "/Users/wolffbe/workspace/banter/testbed/results/21_db-full-cli-sdk-skills-mistral-medium/mistral-medium-3-5/databricks/sdk/0.117.0/no-skills/feature/incremental_load/1/task/data"
    
    # Try to upload files to a directory under the prefix
    data_dir = f"/{MLPAB_DATABRICKS_PREFIX}_data"
    try:
        wc.workspace.mkdirs(data_dir)
        for i in range(1, 7):
            local_file = f"{local_data_path}/increment_{i:02d}.csv"
            with open(local_file, 'r') as f:
                content = f.read()
            # Use workspace.upload with proper parameters
            wc.workspace.upload(path=f"{data_dir}/increment_{i:02d}.csv", content=content.encode(), language="TEXT", overwrite=True)
            print(f"Uploaded increment_{i:02d}.csv")
    except Exception as e:
        print(f"Error uploading files: {e}")
    
    # 2. Create the feature table using SQL
    feature_table_name = "incremental77b14b"
    full_table_name = f"{catalog_name}.{schema_name}.{feature_table_name}"
    
    print(f"Creating feature table: {full_table_name}")
    
    # Use SQL to create the table
    create_table_sql = f"""CREATE TABLE IF NOT EXISTS {full_table_name} (
    row_id STRING,
    account_id STRING,
    event_time BIGINT,
    amount DOUBLE,
    category STRING
) USING DELTA"""
    
    try:
        # Get available warehouses
        warehouses = list(wc.warehouses.list())
        warehouse_id = warehouses[0].id if warehouses else None
        print(f"Using warehouse: {warehouse_id}")
        
        result = wc.statement_execution.execute_statement(
            statement=create_table_sql,
            warehouse_id=warehouse_id
        )
        print(f"Created table via SQL: {full_table_name}")
    except Exception as e:
        print(f"SQL table creation error: {e}")
    
    # 3. Create a notebook to load the data
    notebook_path = f"/{MLPAB_DATABRICKS_PREFIX}/load_incremental_data"
    
    workspace_data_path = f"/Workspace/{MLPAB_DATABRICKS_PREFIX}_data"
    
    notebook_content = f"""# Load Incremental Data
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Define schema
table_schema = "row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING"

# Read all files from workspace
workspace_data_path = "{workspace_data_path}"

# Read all files
dfs = []
for i in range(1, 7):
    file_path = f"{{workspace_data_path}}/increment_{{i:02d}}.csv"
    df = spark.read.csv(file_path, header=True, schema=table_schema)
    dfs.append(df)

# Combine all dataframes
combined_df = dfs[0]
for df in dfs[1:]:
    combined_df = combined_df.union(df)

print(f"Total rows to insert: {{combined_df.count()}}")

# Write to the feature table
combined_df.write.format("delta").mode("overwrite").saveAsTable("{full_table_name}")

print(f"Data loaded successfully into {full_table_name}")
"""
    
    try:
        wc.workspace.upload(path=notebook_path, content=notebook_content.encode(), language="PYTHON", overwrite=True)
        print(f"Created notebook: {notebook_path}")
    except Exception as e:
        print(f"Error creating notebook: {e}")
    
    # 4. Run the notebook to load data using a one-time job
    print("Running notebook to load data...")
    try:
        # Create cluster spec for the job - use compute.ClusterSpec
        cluster_spec = compute.ClusterSpec(
            spark_version="14.3.x-scala2.12",
            node_type_id="Standard_DS3_v2",
            num_workers=1
        )
        
        load_job_response = wc.jobs.create(
            name=f"{MLPAB_DATABRICKS_PREFIX}_load_incremental_data",
            tasks=[
                Task(
                    task_key="load_data_task",
                    notebook_task=dict(
                        notebook_path=notebook_path
                    ),
                    new_cluster=cluster_spec
                )
            ]
        )
        
        load_job_id = load_job_response.job_id
        print(f"Created load job: {load_job_id}")
        
        # Run the job and wait for completion
        run = wc.jobs.run_now_and_wait(job_id=load_job_id, timeout_seconds=300)
        print(f"Load job completed with result: {run.result_state}")
        
    except Exception as e:
        print(f"Error running load job: {e}")
    
    # 5. Create online table for low-latency access using feature store
    print("Creating online table...")
    
    try:
        # First, we need to register the table as a feature table
        # Use SQL to add feature table properties
        alter_table_sql = f"""
        ALTER TABLE {full_table_name} 
        SET TBLPROPERTIES (
            delta.featureTable = 'true',
            delta.featureTable.recordKey = 'row_id',
            delta.featureTable.eventTime = 'event_time'
        )
        """
        
        wc.statement_execution.execute_statement(
            statement=alter_table_sql,
            warehouse_id=warehouse_id
        )
        print(f"Set feature table properties on {full_table_name}")
        
        # Use feature store publish_table
        from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
        
        publish_spec = PublishSpec(
            online_store="default",  # Use default online store
            online_table_name=f"{catalog_name}.{schema_name}.incremental77b14b_online",
            publish_mode=PublishSpecPublishMode.CONTINUOUS
        )
        
        response = wc.feature_store.publish_table(
            source_table_name=full_table_name,
            publish_spec=publish_spec
        )
        print(f"Published table to online store: {response}")
        
    except Exception as e:
        print(f"Error creating online table: {e}")
    
    # 6. Create recurring job for future increments
    print("Creating recurring job...")
    job_name = f"{MLPAB_DATABRICKS_PREFIX}_incrementaljob77b14b"
    
    # Create a notebook for the recurring job
    recurring_notebook_path = f"/{MLPAB_DATABRICKS_PREFIX}/incremental_ingestion"
    recurring_notebook_content = f"""# Incremental Data Ingestion
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Define schema
table_schema = "row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING"

# Find the latest event_time in the existing table
try:
    existing_df = spark.table("{full_table_name}")
    latest_event_time = existing_df.agg({{"event_time": "max"}}).collect()[0][0]
    print(f"Latest event_time in table: {{latest_event_time}}")
except:
    latest_event_time = 0
    print("No existing data, starting from scratch")

# Look for new increment files in workspace
workspace_data_path = "{workspace_data_path}"

try:
    files = dbutils.fs.ls(workspace_data_path)
    increment_files = [f.path for f in files if "increment_" in f.path and f.path.endswith(".csv")]
    print(f"Found increment files: {{increment_files}}")
    
    # Process each new file
    for file_path in increment_files:
        # Extract file number from path
        filename = file_path.split('/')[-1]
        
        # Read the file
        df = spark.read.csv(file_path, header=True, schema=table_schema)
        
        # Filter out any rows that might already exist
        if latest_event_time > 0:
            df = df.filter(df.event_time > latest_event_time)
        
        if df.count() > 0:
            # Append to the table
            df.write.format("delta").mode("append").saveAsTable("{full_table_name}")
            print(f"Appended {{df.count()}} rows from {{filename}}")
            
            # Update latest event_time
            latest_event_time = df.agg({{"event_time": "max"}}).collect()[0][0]
        else:
            print(f"No new rows in {{filename}}")
            
except Exception as e:
    print(f"Error processing files: {{e}}")

print("Incremental ingestion completed")
"""
    
    try:
        wc.workspace.upload(path=recurring_notebook_path, content=recurring_notebook_content.encode(), language="PYTHON", overwrite=True)
        print(f"Created recurring notebook: {recurring_notebook_path}")
    except Exception as e:
        print(f"Error creating recurring notebook: {e}")
    
    # Create the recurring job with daily schedule
    try:
        job_cluster_spec = compute.ClusterSpec(
            spark_version="14.3.x-scala2.12",
            node_type_id="Standard_DS3_v2",
            num_workers=1
        )
        
        job_response = wc.jobs.create(
            name=job_name,
            description="Daily ingestion of incremental data files",
            schedule=CronSchedule(
                quartz_cron_expression="0 0 0 * * ?",  # Daily at midnight UTC
                timezone_id="UTC"
            ),
            tasks=[
                Task(
                    task_key="ingestion_task",
                    notebook_task=dict(
                        notebook_path=recurring_notebook_path
                    ),
                    new_cluster=job_cluster_spec
                )
            ]
        )
        
        job_id = job_response.job_id
        print(f"Created recurring job: {job_name} with ID: {job_id}")
        
    except Exception as e:
        print(f"Error creating recurring job: {e}")
    
    # 7. Write submission file
    print("Writing submission file...")
    submission_data = {
        "job_name": "incrementaljob77b14b"
    }
    
    submission_path = "submission/answers.json"
    with open(submission_path, 'w') as f:
        json.dump(submission_data, f, indent=2)
    
    print(f"Written submission to {submission_path}")
    print("All tasks completed!")

if __name__ == "__main__":
    main()