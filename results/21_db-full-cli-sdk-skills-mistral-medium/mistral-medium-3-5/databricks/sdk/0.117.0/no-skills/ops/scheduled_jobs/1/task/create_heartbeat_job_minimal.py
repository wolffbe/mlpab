#!/usr/bin/env python3
"""Create a recurring heartbeat job on Databricks platform."""

import os
import json
import databricks.sdk
from databricks.sdk.service import jobs

# Environment variables
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabe27f03")

# Job name as specified in the task
JOB_NAME = "heartbeatd0d7ba"

def main():
    # Initialize workspace client
    w = databricks.sdk.WorkspaceClient()
    
    # Get the warehouse ID
    warehouses = list(w.warehouses.list())
    warehouse_id = warehouses[0].id if warehouses else None
    print(f"Using warehouse ID: {warehouse_id}")
    
    if not warehouse_id:
        print("No warehouse found, cannot create job")
        return
    
    # Create the job with a recurring schedule using SQL task
    # Use the same pattern as existing working jobs
    print(f"Creating job: {JOB_NAME}")
    try:
        create_response = w.jobs.create(
            name=JOB_NAME,
            description="Heartbeat job that prints a heartbeat line periodically",
            tags={"purpose": "heartbeat"},
            tasks=[
                jobs.Task(
                    task_key="heartbeat_task",
                    description="Run heartbeat SQL query",
                    sql_task=jobs.SqlTask(
                        warehouse_id=warehouse_id,
                        query=jobs.SqlTaskQuery(query_id="")  # Empty query_id like existing jobs
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
        job_id = create_response.job_id
        print(f"Job created with ID: {job_id}")
        
        # Trigger one run immediately
        print("Triggering first run...")
        run_response = w.jobs.run_now(job_id)
        run_id = run_response.run_id
        print(f"Run triggered with ID: {run_id}")
        
        # Wait for the run to complete
        print("Waiting for run to complete...")
        try:
            final_run = w.jobs.wait_get_run_job_terminated_or_skipped(run_id, timeout_seconds=300)
            print(f"Run completed with state: {final_run.state}")
        except Exception as e:
            print(f"Error waiting for run: {e}")
        
    except Exception as e:
        print(f"Error creating job: {e}")
        print("Trying minimal job creation...")
        # Try with minimal parameters
        try:
            create_response = w.jobs.create(
                name=JOB_NAME,
                tasks=[
                    jobs.Task(
                        task_key="heartbeat_task",
                        sql_task=jobs.SqlTask(
                            warehouse_id=warehouse_id,
                            query=jobs.SqlTaskQuery(query_id="")
                        ),
                    )
                ],
                schedule=jobs.CronSchedule(
                    quartz_cron_expression="0 0 0 * * ?",
                    timezone_id="UTC",
                    pause_status=jobs.PauseStatus.UNPAUSED
                ),
            )
            job_id = create_response.job_id
            print(f"Minimal job created with ID: {job_id}")
            
            # Trigger one run immediately
            print("Triggering first run...")
            run_response = w.jobs.run_now(job_id)
            run_id = run_response.run_id
            print(f"Run triggered with ID: {run_id}")
            
            # Wait for the run to complete
            print("Waiting for run to complete...")
            try:
                final_run = w.jobs.wait_get_run_job_terminated_or_skipped(run_id, timeout_seconds=300)
                print(f"Run completed with state: {final_run.state}")
            except Exception as e:
                print(f"Error waiting for run: {e}")
                
        except Exception as e2:
            print(f"Error creating minimal job: {e2}")
    
    # Create submission directory and write answers.json
    os.makedirs("submission", exist_ok=True)
    submission_data = {
        "job_name": JOB_NAME
    }
    
    with open("submission/answers.json", "w") as f:
        json.dump(submission_data, f, indent=2)
    
    print(f"Submission file written: submission/answers.json")
    print(f"Job '{JOB_NAME}' creation process completed!")

if __name__ == "__main__":
    main()