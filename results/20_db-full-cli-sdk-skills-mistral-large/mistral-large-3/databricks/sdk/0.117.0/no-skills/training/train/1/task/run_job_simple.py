"""Run the training script as a Databricks job and load the predictions into a feature table."""
import os
import time
import requests
from databricks.sdk import WorkspaceClient

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")
JOB_NAME = f"{PREFIX}_trainjobac536a"
TABLE_NAME = "predictionsac536a"
USER = os.getenv("USER", "unknown")
HOST = os.getenv("DATABRICKS_HOST").strip()
if not HOST.startswith(('http://', 'https://')):
    HOST = f"https://{HOST}"
TOKEN = os.getenv("DATABRICKS_TOKEN")

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

# Step 2: Create the job using the REST API
def create_job(workspace_path):
    url = f"{HOST}/api/2.1/jobs/create"
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "name": JOB_NAME,
        "environments": [
            {
                "environment_key": "default",
                "spec": {
                    "client": "1",
                },
            }
        ],
        "tasks": [
            {
                "task_key": "train",
                "environment_key": "default",
                "spark_python_task": {
                    "python_file": workspace_path,
                },
                "timeout_seconds": 3600,
            }
        ],
        "timeout_seconds": 3600,
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if not response.ok:
        print(f"Error creating job: {response.text}")
    response.raise_for_status()
    return response.json()["job_id"]

# Step 3: Run the job and wait for completion
def run_job(job_id):
    url = f"{HOST}/api/2.1/jobs/run-now"
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "job_id": job_id,
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    run_id = response.json()["run_id"]
    
    # Wait for the run to complete
    while True:
        url = f"{HOST}/api/2.1/jobs/runs/get"
        params = {"run_id": run_id}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        run_status = response.json()
        
        if run_status["state"]["life_cycle_state"] in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
            break
        time.sleep(10)
    
    if run_status["state"]["result_state"] != "SUCCESS":
        raise Exception(f"Job failed with state: {run_status['state']['result_state']}")
    
    return run_status

# Step 4: Load predictions.csv into a feature table
def create_feature_table(run_id):
    # Get the run output
    url = f"{HOST}/api/2.1/jobs/runs/get-output"
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    params = {"run_id": run_id}
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    predictions_path = response.json()["logs"]
    
    # Create a feature table
    url = f"{HOST}/api/2.0/mlflow/feature-tables/create"
    payload = {
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
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    # Load data into the feature table
    url = f"{HOST}/api/2.0/mlflow/feature-tables/ingest"
    payload = {
        "feature_table_name": f"{SCHEMA}.{TABLE_NAME}",
        "source_path": predictions_path,
        "source_type": "csv",
        "format_options": {
            "header": "true",
            "inferSchema": "true"
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

# Main workflow
if __name__ == "__main__":
    # Upload the training script to the workspace
    workspace_path = upload_script()
    
    # Create the job
    job_id = create_job(workspace_path)
    
    # Run the job
    run_status = run_job(job_id)
    
    # Create the feature table
    create_feature_table(run_status["run_id"])
    
    # Write the submission file
    submission = {"job_name": JOB_NAME}
    os.makedirs("submission", exist_ok=True)
    with open("submission/answers.json", "w") as f:
        import json
        json.dump(submission, f)
    
    print("Task completed successfully.")