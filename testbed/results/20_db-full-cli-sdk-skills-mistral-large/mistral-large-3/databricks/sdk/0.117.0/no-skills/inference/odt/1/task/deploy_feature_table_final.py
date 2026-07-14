#!/usr/bin/env python3
"""
Deploy a feature table named `scoreda4f6e2`, version 1, in the specified schema.
Columns: request_id, account_id, distance_deg, score.
Enable online access for low-latency lookup.

Steps:
1. Upload input files (requests.csv, profiles.csv) to DBFS.
2. Create a Databricks notebook to compute features and create the table.
3. Deploy a job to run the notebook.
4. Wait for the job to complete.
5. Enable online access.
"""

import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, workspace
from databricks.sdk.service.catalog import *

# Initialize Databricks WorkspaceClient
w = WorkspaceClient()

# Schema and table details
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # e.g., workspace.mlpabceaa54
catalog_name, schema_name = schema_name.split(".")
table_name = "scoreda4f6e2"
full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

# Prefix for all resources
prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")  # e.g., mlpabceaa54

# Upload input files to DBFS
def upload_to_dbfs(local_path, dbfs_path):
    with open(local_path, "rb") as f:
        content = f.read()
    w.dbfs.mkdirs(f"dbfs:/{dbfs_path.rsplit('/', 1)[0]}")
    w.dbfs.put(f"dbfs:/{dbfs_path}", content, overwrite=True)
    print(f"Uploaded {local_path} to dbfs:/{dbfs_path}")

upload_to_dbfs("data/requests.csv", f"{prefix}_data/requests.csv")
upload_to_dbfs("data/profiles.csv", f"{prefix}_data/profiles.csv")

# Create a notebook to compute features and create the table
notebook_content = f"""# Databricks notebook source
# MAGIC %md
# MAGIC ## Feature Table: {table_name}
# MAGIC 
This notebook creates a feature table named `{table_name}` with columns: `request_id`, `account_id`, `distance_deg`, `score`.

**Schema:** `{schema_name}`

**Source Data:**
- `dbfs:/{prefix}_data/requests.csv` (requests)
- `dbfs:/{prefix}_data/profiles.csv` (profiles)

**Transformations:**
- `distance_deg = sqrt((request_lat - home_lat)^2 + (request_lon - home_lon)^2)` (rounded to 6 decimal places)
- `score = base_score - 0.1 * distance_deg` (rounded to 6 decimal places)

**Output:**
- Feature table: `{table_name}`

---

# COMMAND ----------

from pyspark.sql.functions import col, sqrt, round

# Read source data
requests_df = spark.read.csv("dbfs:/{prefix}_data/requests.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv("dbfs:/{prefix}_data/profiles.csv", header=True, inferSchema=True)

# Join requests with profiles
merged_df = requests_df.join(profiles_df, "account_id", "left")

# Compute on-demand features
merged_df = merged_df.withColumn(
    "distance_deg", 
    round(
        sqrt((col("request_lat") - col("home_lat"))**2 + (col("request_lon") - col("home_lon"))**2),
        6
    )
)

merged_df = merged_df.withColumn(
    "score", 
    round(col("base_score") - 0.1 * col("distance_deg"), 6)
)

# Select required columns
result_df = merged_df.select("request_id", "account_id", "distance_deg", "score")

# Write to the feature table
result_df.write.saveAsTable("{full_table_name}", mode="overwrite")

print(f"Feature table {full_table_name} created successfully.")
"""

# Upload the notebook to the workspace
notebook_path = f"/Users/{os.getenv('USER')}/{prefix}/feature_notebook"
w.workspace.mkdirs(notebook_path)
w.workspace.upload(notebook_path + "/feature_notebook", notebook_content.encode(), format=workspace.ImportFormat.SOURCE)
print(f"Notebook uploaded to {notebook_path}")

# Create a job to run the notebook
job_name = f"{prefix}_feature_job"
job = w.jobs.create(
    name=job_name,
    tasks=[
        jobs.Task(
            task_key="feature_task",
            notebook_task=jobs.NotebookTask(
                notebook_path=notebook_path + "/feature_notebook",
            ),
            existing_cluster_id=w.clusters.list()[0].cluster_id,
        )
    ],
)
print(f"Job {job_name} created with ID: {job.job_id}")

# Run the job
run = w.jobs.run_now(job_id=job.job_id)
print(f"Job started with run ID: {run.run_id}")

# Wait for the job to complete
print("Waiting for job to complete...")
while True:
    run_status = w.jobs.get_run(run_id=run.run_id)
    if run_status.state.life_cycle_state == jobs.RunLifeCycleState.TERMINATED:
        if run_status.state.result_state == jobs.RunResultState.SUCCESS:
            print("Job completed successfully.")
            break
        else:
            raise Exception(f"Job failed: {run_status.state.state_message}")
    time.sleep(10)

# Enable online access for low-latency lookup
try:
    w.online_tables.create(
        name=table_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        primary_key_columns=["request_id"],
        source_table_full_name=full_table_name,
    )
    print(f"Online table {full_table_name} enabled for low-latency access.")
except Exception as e:
    print(f"Online table creation failed: {e}")
    raise

print(f"Feature table {full_table_name} created successfully with online access enabled.")