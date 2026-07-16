#!/usr/bin/env python3
"""
Submits the churn_training_job.py as a Databricks job.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

# Environment variables
SCHEMA_NAME = os.environ["MLPAB_DATABRICKS_SCHEMA"].replace(".", "_")
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]

# Initialize WorkspaceClient
w = WorkspaceClient()

# Upload the job script to DBFS
def upload_job_script():
    try:
        # Upload to DBFS
        dbfs_path = f"/tmp/{PREFIX}_churn_training_job.py"
        with open("./churn_training_job.py", "rb") as f:
            w.dbfs.upload(dbfs_path, f, overwrite=True)
        print(f"Uploaded job script to DBFS: {dbfs_path}")
        return dbfs_path
    except Exception as e:
        print(f"Error uploading job script: {e}")
        return None

# Submit the job
def submit_job(dbfs_path):
    try:
        # Get the first available cluster
        clusters = list(w.clusters.list())
        if not clusters:
            raise Exception("No clusters available")
        cluster_id = clusters[0].cluster_id
        
        # Create the job
        job = w.jobs.create(
            name=f"{PREFIX}_churn_training_job",
            tasks=[
                jobs.PythonWheelTask(
                    package_name="churn_training",
                    entry_point="main"
                )
                if False else  # Skip PythonWheelTask, use SparkPythonTask
                jobs.SparkPythonTask(
                    python_file=f"dbfs:{dbfs_path}",
                    parameters=[f"--schema={SCHEMA_NAME}"]
                )
            ],
            existing_cluster_id=cluster_id,
            email_notifications=jobs.JobEmailNotifications(
                on_success=[],
                on_failure=[]
            ),
            timeout_seconds=3600
        )
        
        print(f"Created job: {job.job_id}")
        
        # Run the job
        run = w.jobs.run_now(job_id=job.job_id)
        print(f"Started job run: {run.run_id}")
        
        # Wait for completion
        run_result = w.jobs.wait_get_run_job_terminated_or_skipped(run_id=run.run_id)
        
        if run_result.result_state == jobs.RunResultState.SUCCESS:
            print("Job completed successfully.")
        else:
            print(f"Job failed: {run_result.state}")
        
    except Exception as e:
        print(f"Error submitting job: {e}")

if __name__ == "__main__":
    dbfs_path = upload_job_script()
    if dbfs_path:
        submit_job(dbfs_path)