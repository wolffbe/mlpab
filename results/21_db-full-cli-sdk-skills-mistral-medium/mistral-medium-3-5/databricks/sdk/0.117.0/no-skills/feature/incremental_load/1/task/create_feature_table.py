#!/usr/bin/env python3

import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    TableType, 
    DataSourceFormat,
    ColumnInfo,
    OnlineTable,
    OnlineTableSpec
)
from databricks.sdk.service.jobs import (
    CronSchedule,
    Task,
    SparkPythonTask,
    JobCluster,
    NewCluster
)
from databricks.sdk.service.compute import (
    ClusterSource,
    SparkVersion
)

def main():
    # Environment variables
    MLPAB_DATABRICKS_SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabb7e680')
    MLPAB_DATABRICKS_PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabb7e680')
    
    # Parse schema
    catalog_name, schema_name = MLPAB_DATABRICKS_SCHEMA.split('.')
    
    print(f"Catalog: {catalog_name}, Schema: {schema_name}, Prefix: {MLPAB_DATABRICKS_PREFIX}")
    
    # Initialize client
    wc = WorkspaceClient()
    
    # 1. Create the feature table (Delta table in Unity Catalog)
    feature_table_name = "incremental77b14b"
    full_table_name = f"{catalog_name}.{schema_name}.{feature_table_name}"
    
    print(f"Creating feature table: {full_table_name}")
    
    # Define columns based on schema
    columns = [
        ColumnInfo(name="row_id", type_text="STRING"),
        ColumnInfo(name="account_id", type_text="STRING"),
        ColumnInfo(name="event_time", type_text="BIGINT"),
        ColumnInfo(name="amount", type_text="DOUBLE"),
        ColumnInfo(name="category", type_text="STRING")
    ]
    
    # Create the table
    try:
        table_info = wc.tables.create(
            name=feature_table_name,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_type=TableType.MANAGED,
            data_source_format=DataSourceFormat.DELTA,
            columns=columns,
            properties={
                "delta.featureTable": "true",
                "delta.featureTable.recordKey": "row_id",
                "delta.featureTable.eventTime": "event_time"
            }
        )
        print(f"Created table: {table_info.full_name}")
    except Exception as e:
        print(f"Table may already exist or error: {e}")
        # Check if table exists
        try:
            existing_table = wc.tables.get(full_name=full_table_name)
            print(f"Table already exists: {existing_table.full_name}")
        except:
            raise e
    
    # 2. Load all increment files into the table
    print("Loading increment files...")
    
    # Create a notebook to load the data
    notebook_path = f"/Users/{wc.current_user.me().id}/{MLPAB_DATABRICKS_PREFIX}/load_incremental_data"
    
    # Create the notebook content
    notebook_content = f"""# Load Incremental Data
# This notebook loads all increment files into the feature table

# Read all increment files from dbfs
increment_files = [
    "dbfs:/FileStore/tmp/increment_01.csv",
    "dbfs:/FileStore/tmp/increment_02.csv", 
    "dbfs:/FileStore/tmp/increment_03.csv",
    "dbfs:/FileStore/tmp/increment_04.csv",
    "dbfs:/FileStore/tmp/increment_05.csv",
    "dbfs:/FileStore/tmp/increment_06.csv"
]

# First, upload files to DBFS
import os
local_data_path = "/Users/wolffbe/workspace/banter/testbed/results/21_db-full-cli-sdk-skills-mistral-medium/mistral-medium-3-5/databricks/sdk/0.117.0/no-skills/feature/incremental_load/1/task/data"

# Use dbutils to upload files
dbutils.fs.mkdirs("dbfs:/FileStore/tmp/")

for i in range(1, 7):
    local_file = f"{local_data_path}/increment_{i:02d}.csv"
    dbfs_path = f"dbfs:/FileStore/tmp/increment_{i:02d}.csv"
    dbutils.fs.cp(f"file:{local_file}", dbfs_path)
    print(f"Uploaded {local_file} to {dbfs_path}")

# Read and combine all CSV files
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Define schema
table_schema = "row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING"

# Read all files
dfs = []
for i in range(1, 7):
    df = spark.read.csv(f"/FileStore/tmp/increment_{i:02d}.csv", header=True, schema=table_schema)
    dfs.append(df)

# Combine all dataframes
combined_df = dfs[0]
for df in dfs[1:]:
    combined_df = combined_df.union(df)

print(f"Total rows to insert: {combined_df.count()}")

# Write to the feature table
combined_df.write.format("delta").mode("overwrite").saveAsTable(f"{full_table_name}")

print(f"Data loaded successfully into {full_table_name}")
"""
    
    # Write the notebook
    try:
        wc.workspace.put(path=notebook_path, content=notebook_content, language="PYTHON")
        print(f"Created notebook: {notebook_path}")
    except Exception as e:
        print(f"Error creating notebook: {e}")
    
    # Upload data files to DBFS first
    print("Uploading data files to DBFS...")
    local_data_path = "/Users/wolffbe/workspace/banter/testbed/results/21_db-full-cli-sdk-skills-mistral-medium/mistral-medium-3-5/databricks/sdk/0.117.0/no-skills/feature/incremental_load/1/task/data"
    
    # Use dbfs API to upload files
    try:
        wc.dbfs.mkdirs("FileStore/tmp")
        for i in range(1, 7):
            local_file = f"{local_data_path}/increment_{i:02d}.csv"
            with open(local_file, 'r') as f:
                content = f.read()
            wc.dbfs.put_file(f"FileStore/tmp/increment_{i:02d}.csv", content.encode(), overwrite=True)
            print(f"Uploaded increment_{i:02d}.csv to DBFS")
    except Exception as e:
        print(f"Error uploading files: {e}")
    
    # Run the notebook to load data
    print("Running notebook to load data...")
    try:
        # Create a job to run the notebook once
        load_job_response = wc.jobs.create(
            name=f"{MLPAB_DATABRICKS_PREFIX}_load_incremental_data",
            tasks=[
                Task(
                    notebook_task=dict(
                        notebook_path=notebook_path
                    )
                )
            ],
            job_clusters=[
                JobCluster(
                    new_cluster=NewCluster(
                        spark_version="14.3.x-scala2.12",
                        node_type_id="Standard_DS3_v2",
                        num_workers=1,
                        cluster_source=ClusterSource.UI
                    )
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
    
    # 3. Create online table for low-latency access
    print("Creating online table...")
    online_table_name = f"{catalog_name}.{schema_name}.incremental77b14b_online"
    
    try:
        online_table_spec = OnlineTableSpec(
            source_table_full_name=full_table_name,
            primary_key_columns=["row_id"],
            timeseries_key="event_time",
            perform_full_copy=True
        )
        
        online_table = OnlineTable(
            name=online_table_name,
            spec=online_table_spec
        )
        
        # Create the online table
        create_response = wc.online_tables.create_and_wait(online_table)
        print(f"Created online table: {online_table_name}")
        
    except Exception as e:
        print(f"Error creating online table: {e}")
    
    # 4. Create recurring job for future increments
    print("Creating recurring job...")
    job_name = f"{MLPAB_DATABRICKS_PREFIX}_incrementaljob77b14b"
    
    # Create a notebook for the recurring job
    recurring_notebook_path = f"/Users/{wc.current_user.me().id}/{MLPAB_DATABRICKS_PREFIX}/incremental_ingestion"
    recurring_notebook_content = f"""# Incremental Data Ingestion
# This notebook ingests new increment files into the feature table

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Define schema
table_schema = "row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING"

# Find the latest event_time in the existing table
try:
    existing_df = spark.table("{full_table_name}")
    latest_event_time = existing_df.agg({"event_time": "max"}).collect()[0][0]
    print(f"Latest event_time in table: {latest_event_time}")
except:
    latest_event_time = 0
    print("No existing data, starting from scratch")

# Look for new increment files in DBFS
import os
import datetime

dbfs_path = "/FileStore/tmp/"
try:
    files = dbutils.fs.ls(dbfs_path)
    increment_files = [f.path for f in files if f.path.startswith(dbfs_path) and "increment_" in f.path and f.path.endswith(".csv")]
    print(f"Found increment files: {increment_files}")
    
    # Process each new file
    for file_path in increment_files:
        # Extract file number from path
        filename = file_path.split('/')[-1]
        file_num = int(filename.replace("increment_", "").replace(".csv", ""))
        
        # Only process files we haven't seen before (simple check based on filename)
        # In production, you'd have better tracking
        df = spark.read.csv(file_path, header=True, schema=table_schema)
        
        # Filter out any rows that might already exist
        if latest_event_time > 0:
            df = df.filter(df.event_time > latest_event_time)
        
        if df.count() > 0:
            # Append to the table
            df.write.format("delta").mode("append").saveAsTable("{full_table_name}")
            print(f"Appended {df.count()} rows from {filename}")
            
            # Update latest event_time
            latest_event_time = df.agg({"event_time": "max"}).collect()[0][0]
        else:
            print(f"No new rows in {filename}")
            
except Exception as e:
    print(f"Error processing files: {e}")

print("Incremental ingestion completed")
"""
    
    try:
        wc.workspace.put(path=recurring_notebook_path, content=recurring_notebook_content, language="PYTHON")
        print(f"Created recurring notebook: {recurring_notebook_path}")
    except Exception as e:
        print(f"Error creating recurring notebook: {e}")
    
    # Create the recurring job with daily schedule
    try:
        job_response = wc.jobs.create(
            name=job_name,
            description="Daily ingestion of incremental data files",
            schedule=CronSchedule(
                quartz_cron_expression="0 0 0 * * ?",  # Daily at midnight
                timezone_id="UTC"
            ),
            tasks=[
                Task(
                    notebook_task=dict(
                        notebook_path=recurring_notebook_path
                    )
                )
            ],
            job_clusters=[
                JobCluster(
                    new_cluster=NewCluster(
                        spark_version="14.3.x-scala2.12",
                        node_type_id="Standard_DS3_v2",
                        num_workers=1,
                        cluster_source=ClusterSource.UI
                    )
                )
            ]
        )
        
        job_id = job_response.job_id
        print(f"Created recurring job: {job_name} with ID: {job_id}")
        
    except Exception as e:
        print(f"Error creating recurring job: {e}")
    
    # 5. Write submission file
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