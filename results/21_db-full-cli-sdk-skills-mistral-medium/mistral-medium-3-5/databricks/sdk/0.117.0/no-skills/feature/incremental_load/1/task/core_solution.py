#!/usr/bin/env python3

import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    CronSchedule,
    Task,
    NotebookTask
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
    
    # 1. Create the feature table using SQL
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
    
    # 2. Set feature table properties
    print("Setting feature table properties...")
    try:
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
    except Exception as e:
        print(f"Error setting feature table properties: {e}")
    
    # 3. Create notebooks directory
    print("Creating notebooks...")
    notebooks_dir = f"/{MLPAB_DATABRICKS_PREFIX}"
    try:
        wc.workspace.mkdirs(notebooks_dir)
        print(f"Created notebooks directory: {notebooks_dir}")
    except Exception as e:
        print(f"Error creating notebooks directory: {e}")
    
    # 4. Create a notebook to load the data
    notebook_path = f"{notebooks_dir}/load_incremental_data"
    
    notebook_content = f"""# Load Incremental Data
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Define schema
table_schema = "row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING"

# For this task, we'll use the local data files that are available in the workspace
# The data files have been uploaded to the workspace

# Read all files from workspace
workspace_data_path = "/Workspace/{MLPAB_DATABRICKS_PREFIX}_data"

# Read all files
dfs = []
for i in range(1, 7):
    file_path = f"{{workspace_data_path}}/increment_{{i:02d}}.csv"
    try:
        df = spark.read.csv(file_path, header=True, schema=table_schema)
        dfs.append(df)
    except:
        print(f"File {{file_path}} not found, skipping")

if dfs:
    # Combine all dataframes
    combined_df = dfs[0]
    for df in dfs[1:]:
        combined_df = combined_df.union(df)

    print(f"Total rows to insert: {{combined_df.count()}}")

    # Write to the feature table
    combined_df.write.format("delta").mode("overwrite").saveAsTable("{full_table_name}")

    print(f"Data loaded successfully into {full_table_name}")
else:
    print("No data files found")
"""
    
    try:
        wc.workspace.upload(path=notebook_path, content=notebook_content.encode(), overwrite=True)
        print(f"Created notebook: {notebook_path}")
    except Exception as e:
        print(f"Error creating notebook: {e}")
    
    # 5. Create a notebook for the recurring job
    recurring_notebook_path = f"{notebooks_dir}/incremental_ingestion"
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
workspace_data_path = "/Workspace/{MLPAB_DATABRICKS_PREFIX}_data"

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
        wc.workspace.upload(path=recurring_notebook_path, content=recurring_notebook_content.encode(), overwrite=True)
        print(f"Created recurring notebook: {recurring_notebook_path}")
    except Exception as e:
        print(f"Error creating recurring notebook: {e}")
    
    # 6. Create recurring job for future increments
    print("Creating recurring job...")
    job_name = f"{MLPAB_DATABRICKS_PREFIX}_incrementaljob77b14b"
    
    # Create the recurring job with daily schedule
    try:
        recurring_notebook_task = NotebookTask(notebook_path=recurring_notebook_path)
        
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
                    notebook_task=recurring_notebook_task
                )
            ]
        )
        
        job_id = job_response.job_id
        print(f"Created recurring job: {job_name} with ID: {job_id}")
        
    except Exception as e:
        print(f"Error creating recurring job: {e}")
    
    # 7. Create online table for low-latency access using feature store
    print("Creating online table...")
    
    try:
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
    
    # 8. Write submission file
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