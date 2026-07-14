#!/usr/bin/env python3
"""
Submits the churn_training_job.py as a Databricks job using a workspace file.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, workspace

# Environment variables
SCHEMA_NAME = os.environ["MLPAB_DATABRICKS_SCHEMA"].replace(".", "_")
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
USER = os.environ.get("DATABRICKS_USER", "unknown_user")

# Initialize WorkspaceClient
w = WorkspaceClient()

# Upload the job script to the workspace
def upload_job_script():
    try:
        # Define the workspace path
        workspace_path = f"/Users/{USER}/{PREFIX}_churn_training_job"
        
        # Read the script
        with open("./churn_training_job.py", "rb") as f:
            script_content = f.read()
        
        # Upload to workspace
        w.workspace.upload(workspace_path, script_content, format=workspace.ImportFormat.SOURCE, overwrite=True)
        print(f"Uploaded job script to workspace: {workspace_path}")
        return workspace_path
    except Exception as e:
        print(f"Error uploading job script: {e}")
        return None

# Submit the job
def submit_job(workspace_path):
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
                jobs.SparkPythonTask(
                    python_file=workspace_path,
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
    workspace_path = upload_job_script()
    if workspace_path:
        submit_job(workspace_path)