#!/usr/bin/env python3
"""
Upload input files to a volume and submit a job to build the training dataset.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, jobs

# Initialize Databricks SDK
w = WorkspaceClient()

# Environment variables
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")
prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")
volume_name = f"{prefix}_input_data"
table_name = "churntrainingaf8b21"
job_name = f"{prefix}_build_training_dataset"

# Create a volume to store input files
print(f"Creating volume {schema_name}.{volume_name}...")
w.volumes.create(
    catalog_name=schema_name.split(".")[0],
    schema_name=schema_name.split(".")[1],
    name=volume_name,
    volume_type=catalog.VolumeType.MANAGED,
)

# Upload input files to the volume
input_files = [
    "transactions.csv",
    "profiles.csv", 
    "activity.csv",
    "account_health.csv",
    "transactions_late.csv",
    "labels.csv",
]

volume_path = f"/Volumes/{schema_name}/{volume_name}"
for file_name in input_files:
    local_path = f"./data/{file_name}"
    remote_path = f"{volume_path}/{file_name}"
    print(f"Uploading {local_path} to {remote_path}...")
    w.files.upload(remote_path=remote_path, source_path=local_path)

# Create a job to run the PySpark script
print(f"Creating job {job_name}...")
job = w.jobs.create(
    name=job_name,
    tasks=[
        {
            "task_key": "build_training_dataset",
            "spark_python_task": {
                "python_file": f"{volume_path}/job_script.py",
            },
            "existing_cluster_id": "1201-my-cluster",  # Use an existing cluster
            "libraries": [],
            "spark_python_task": {
                "python_file": f"dbfs:{volume_path}/job_script.py",
            },
            "spark_conf": {
                "spark.databricks.schema.name": schema_name,
                "spark.databricks.volume.name": volume_name,
            },
        }
    ],
)

# Submit the job
print(f"Submitting job {job_name}...")
run = w.jobs.run_now(job_id=job.job_id)

print(f"Job submitted! Run ID: {run.run_id}")
print(f"Training dataset will be written to {schema_name}.{table_name}")