#!/usr/bin/env python3
"""
Upload input files and PySpark script to a volume, then submit a job to build the training dataset.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute

# Initialize Databricks SDK
w = WorkspaceClient()

# Environment variables
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")
catalog_name = schema_name.split(".")[0]
schema = schema_name.split(".")[1]
prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")
volume_name = f"{prefix}_input_data"
table_name = "churntrainingaf8b21"
job_name = f"{prefix}_build_training_dataset"

# Upload the PySpark script to the volume
script_local_path = "./job_script_params.py"
script_remote_path = f"/Volumes/{catalog_name}/{schema}/{volume_name}/job_script.py"
print(f"Uploading {script_local_path} to {script_remote_path}...")
with open(script_local_path, "rb") as f:
    w.files.upload(script_remote_path, f)

# Upload input files to the volume
input_files = [
    "transactions.csv",
    "profiles.csv", 
    "activity.csv",
    "account_health.csv",
    "transactions_late.csv",
    "labels.csv",
]

for file_name in input_files:
    local_path = f"./data/{file_name}"
    remote_path = f"/Volumes/{catalog_name}/{schema}/{volume_name}/{file_name}"
    print(f"Uploading {local_path} to {remote_path}...")
    with open(local_path, "rb") as f:
        w.files.upload(remote_path, f)

# Create a job to run the PySpark script
print(f"Creating job {job_name}...")
new_cluster = compute.ClusterSpec(
    spark_version="13.3.x-scala2.12",
    node_type_id="i3.xlarge",
    num_workers=0,
    spark_conf={
        "spark.databricks.cluster.profile": "singleNode",
        "spark.master": "local[*]",
    },
    custom_tags={
        "ResourceClass": "SingleNode",
    },
)

task = jobs.Task(
    task_key="build_training_dataset",
    new_cluster=new_cluster,
    spark_python_task=jobs.SparkPythonTask(
        python_file=f"dbfs:{script_remote_path}",
        parameters=[f"--schema={schema_name}", f"--volume={volume_name}"],
    ),
    libraries=[],
)

job = w.jobs.create(
    name=job_name,
    tasks=[task],
)

# Submit the job
print(f"Submitting job {job_name}...")
run = w.jobs.run_now(job_id=job.job_id)

print(f"Job submitted! Run ID: {run.run_id}")
print(f"Training dataset will be written to {schema_name}.{table_name}")