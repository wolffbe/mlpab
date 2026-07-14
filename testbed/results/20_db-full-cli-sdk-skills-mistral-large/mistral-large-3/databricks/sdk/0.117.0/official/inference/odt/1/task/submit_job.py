#!/usr/bin/env python3

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

# Initialize Databricks WorkspaceClient
w = WorkspaceClient()

# Define schema and job name
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")
job_name = f"{os.getenv('MLPAB_DATABRICKS_PREFIX')}_feature_table_job"

# Upload input files to DBFS
for file_name in ["requests.csv", "profiles.csv"]:
    file_path = f"data/{file_name}"
    dbfs_path = f"/FileStore/{file_name}"
    
    with open(file_path, "rb") as f:
        w.dbfs.upload(dbfs_path, f, overwrite=True)

# Define the job
job = w.jobs.create(
    name=job_name,
    tasks=[
        jobs.Task(
            task_key="transform_and_write",
            new_cluster=jobs.ClusterSpec(
                spark_version="13.3.x-scala2.12",
                node_type_id="i3.xlarge",
                num_workers=1,
                autoscale=jobs.AutoScale(min_workers=1, max_workers=2)
            ),
            spark_python_task=jobs.SparkPythonTask(
                python_file=f"dbfs:/FileStore/transform_and_write.py"
            ),
            libraries=[
                jobs.Library(pypi=jobs.PythonPyPiLibrary(package="pandas")),
                jobs.Library(pypi=jobs.PythonPyPiLibrary(package="numpy"))
            ]
        )
    ]
)

# Upload the transformation script to DBFS
with open("transform_and_write.py", "rb") as f:
    w.dbfs.upload("/FileStore/transform_and_write.py", f, overwrite=True)

# Run the job
run = w.jobs.run_now(job_id=job.job_id)

print(f"Job {job_name} submitted with run ID: {run.run_id}")