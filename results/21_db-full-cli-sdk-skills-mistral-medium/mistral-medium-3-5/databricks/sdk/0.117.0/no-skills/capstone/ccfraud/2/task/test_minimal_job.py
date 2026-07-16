#!/usr/bin/env python3
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, NotebookTask

wc = WorkspaceClient()
PREFIX = 'mlpabf21a49'
WAREHOUSE_ID = 'a832b544eb7dc3fe'

def main():
    print("Creating job for minimal notebook...")
    
    job_name = f"{PREFIX}_test_minimal"
    
    # Use serverless warehouse
    task = Task(
        task_key="test_minimal",
        notebook_task=NotebookTask(
            notebook_path='/Users/benedict@hopsworks.ai/mlpabf21a49/test_minimal',
            warehouse_id=WAREHOUSE_ID
        )
    )
    
    job = wc.jobs.create(
        name=job_name,
        tasks=[task]
    )
    
    print(f"Job created: {job.job_id}")
    
    # Run the job
    run = wc.jobs.run_now(job.job_id)
    print(f"Job run started: {run.run_id}")
    
    print("Job is running on the platform.")
    print(f"Job ID: {job.job_id}")
    print(f"Run ID: {run.run_id}")

if __name__ == "__main__":
    main()
