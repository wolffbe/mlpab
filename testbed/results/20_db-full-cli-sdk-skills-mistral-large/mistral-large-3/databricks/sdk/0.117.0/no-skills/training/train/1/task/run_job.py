"""Run the training script as a Databricks job and load the predictions into a feature table."""
import os
import time
import base64
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute, ml

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")
JOB_NAME = f"{PREFIX}_trainjobac536a"
TABLE_NAME = "predictionsac536a"
USER = os.getenv("USER", "unknown")

# Initialize the WorkspaceClient
w = WorkspaceClient()

# Step 1: Upload the training script to the workspace
def upload_script():
    script_path = "data/train_model.py"
    workspace_dir = f"/Shared/{PREFIX}"
    workspace_path = f"{workspace_dir}/train_model.py"
    
    # Create the directory if it doesn't exist
    try:
        w.workspace.mkdirs(workspace_dir)
    except Exception as e:
        print(f"Directory creation skipped or failed: {e}")
    
    # Read the script content
    with open(script_path, "r") as f:
        script_content = f.read()
    
    # Upload to workspace
    w.workspace.upload(workspace_path, script_content.encode("utf-8"), overwrite=True)
    return workspace_path

# Step 2: Create the job
def create_job(workspace_path):
    # Define the job for serverless execution with a default environment
    job = w.jobs.create(
        name=JOB_NAME,
        environments=[
            jobs.JobEnvironment(
                environment_key="default",
                spec=compute.Environment(
                    dependencies=[
                        "pandas==2.0.3",
                        "numpy==1.24.3",
                    ],
                ),
            )
        ],
        tasks=[
            jobs.Task(
                task_key="train",
                environment_key="default",
                spark_python_task=jobs.SparkPythonTask(
                    python_file=workspace_path,
                ),
                timeout_seconds=3600,
            )
        ],
        timeout_seconds=3600,
    )
    return job

# Step 3: Run the job and wait for completion
def run_job(job_id):
    run = w.jobs.run_now(job_id=job_id).result()
    
    # Wait for the run to complete
    while True:
        run_status = w.jobs.get_run(run.run_id)
        if run_status.state.life_cycle_state in [jobs.RunLifeCycleState.TERMINATED, jobs.RunLifeCycleState.SKIPPED, jobs.RunLifeCycleState.INTERNAL_ERROR]:
            break
        time.sleep(10)
    
    if run_status.state.result_state != jobs.RunResultState.SUCCESS:
        raise Exception(f"Job failed with state: {run_status.state.result_state}")
    
    return run_status

# Step 4: Load predictions.csv into a feature table
def create_feature_table(run_id):
    # Get the run output
    run_output = w.jobs.get_run_output(run_id)
    predictions_path = run_output.logs
    
    # Create a feature table
    feature_table = w.api_client.do(
        "POST", 
        "/api/2.0/mlflow/feature-tables/create",
        body={
            "name": f"{SCHEMA}.{TABLE_NAME}",
            "description": "Predictions from training job",
            "primary_keys": ["row_id"],
            "features": [
                {
                    "name": "score",
                    "type": "float",
                }
            ],
            "online_store": {
                "enable_online_store": True
            }
        }
    )
    
    # Load data into the feature table
    w.api_client.do(
        "POST", 
        "/api/2.0/mlflow/feature-tables/ingest",
        body={
            "feature_table_name": f"{SCHEMA}.{TABLE_NAME}",
            "source_path": predictions_path,
            "source_type": "csv",
            "format_options": {
                "header": "true",
                "inferSchema": "true"
            }
        }
    )
    
    return feature_table

# Main workflow
if __name__ == "__main__":
    # Upload the training script to the workspace
    workspace_path = upload_script()
    
    # Create the job
    job = create_job(workspace_path)
    job_id = job.job_id
    
    # Run the job
    run_status = run_job(job_id)
    
    # Create the feature table
    create_feature_table(run_status.run_id)
    
    # Write the submission file
    submission = {"job_name": JOB_NAME}
    os.makedirs("submission", exist_ok=True)
    with open("submission/answers.json", "w") as f:
        import json
        json.dump(submission, f)
    
    print("Task completed successfully.")