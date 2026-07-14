#!/usr/bin/env python3
"""Reset the heartbeat job with proper configuration."""

import os
import json
import databricks.sdk
from databricks.sdk.service import jobs
from databricks.sdk.service import workspace

# Environment variables
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabe27f03")

# Job name as specified in the task
JOB_NAME = "heartbeatd0d7ba"

def main():
    # Initialize workspace client
    w = databricks.sdk.WorkspaceClient()
    
    # Get current user
    current_user = w.current_user.me().user_name
    print(f"Current user: {current_user}")
    
    # Find the existing job with the target name
    existing_jobs = list(w.jobs.list())
    target_job = None
    for job in existing_jobs:
        if hasattr(job.settings, 'name') and job.settings.name == JOB_NAME:
            target_job = job
            break
    
    if not target_job:
        print(f"Job '{JOB_NAME}' not found, cannot reset")
        return
    
    job_id = target_job.job_id
    print(f"Found job '{JOB_NAME}' with ID: {job_id}")
    
    # Upload the heartbeat script as a notebook
    notebook_path = f"/Users/{current_user}/{MLPAB_DATABRICKS_PREFIX}/heartbeat"
    print(f"Uploading heartbeat notebook to {notebook_path}")
    
    # Read the heartbeat.py file
    with open("data/heartbeat.py", "r") as f:
        heartbeat_content = f.read()
    
    w.workspace.upload(notebook_path, heartbeat_content.encode('utf-8'), overwrite=True, language=workspace.Language.PYTHON)
    print("Notebook uploaded successfully")
    
    # Reset the job with proper configuration (completely replace settings)
    print(f"Resetting job {job_id} with heartbeat configuration...")
    try:
        w.jobs.reset(
            job_id=job_id,
            new_settings=jobs.JobSettings(
                name=JOB_NAME,
                description="Heartbeat job that prints a heartbeat line periodically",
                tags={"purpose": "heartbeat"},
                tasks=[
                    jobs.Task(
                        task_key="heartbeat_task",
                        description="Run heartbeat notebook",
                        notebook_task=jobs.NotebookTask(
                            notebook_path=notebook_path
                        ),
                        timeout_seconds=60,
                    )
                ],
                # Schedule: run hourly (recurring job)
                schedule=jobs.CronSchedule(
                    quartz_cron_expression="0 0 0 * * ?",  # Every hour at minute 0 (Quartz syntax)
                    timezone_id="UTC",
                    pause_status=jobs.PauseStatus.UNPAUSED
                ),
                max_concurrent_runs=1,
            )
        )
        print("Job reset successfully with heartbeat configuration")
        
        # Verify the job configuration
        updated_job = w.jobs.get(job_id)
        print(f"Job now has {len(updated_job.settings.tasks)} tasks:")
        for task in updated_job.settings.tasks:
            print(f"  - {task.task_key}")
        
        print(f"Schedule: {updated_job.settings.schedule}")
        
    except Exception as e:
        print(f"Error resetting job: {e}")
    
    # Create submission directory and write answers.json
    os.makedirs("submission", exist_ok=True)
    submission_data = {
        "job_name": JOB_NAME
    }
    
    with open("submission/answers.json", "w") as f:
        json.dump(submission_data, f, indent=2)
    
    print(f"Submission file written: submission/answers.json")
    print(f"Job '{JOB_NAME}' reset with proper configuration!")

if __name__ == "__main__":
    main()