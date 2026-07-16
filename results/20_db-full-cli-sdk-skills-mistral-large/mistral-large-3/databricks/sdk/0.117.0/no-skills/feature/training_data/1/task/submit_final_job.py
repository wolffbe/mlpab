#!/usr/bin/env python3
"""
Submits the churn_training_job.py as a Databricks job using the REST API.
"""

import os
import requests
import base64
import json

# Environment variables
SCHEMA_NAME = os.environ["MLPAB_DATABRICKS_SCHEMA"].replace(".", "_")
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
DATABRICKS_HOST = os.environ["DATABRICKS_HOST"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
USER = os.environ.get("DATABRICKS_USER", "unknown_user")

# Headers for API requests
headers = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

# Upload the job script to the workspace
def upload_job_script():
    try:
        # Define the workspace path
        workspace_path = f"/Users/{USER}/{PREFIX}_churn_training_job"
        
        # Read the script
        with open("./churn_training_job.py", "r") as f:
            script_content = f.read()
        
        # Encode the script
        encoded_content = base64.b64encode(script_content.encode("utf-8")).decode("utf-8")
        
        # Upload to workspace
        url = f"{DATABRICKS_HOST}/api/2.0/workspace/import"
        payload = {
            "path": workspace_path,
            "content": encoded_content,
            "language": "PYTHON",
            "overwrite": True,
            "format": "SOURCE"
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            raise Exception(f"Failed to upload script: {response.text}")
        
        print(f"Uploaded job script to workspace: {workspace_path}")
        return workspace_path
    except Exception as e:
        print(f"Error uploading job script: {e}")
        return None

# Get the first available cluster
def get_cluster_id():
    try:
        url = f"{DATABRICKS_HOST}/api/2.0/clusters/list"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to list clusters: {response.text}")
        
        clusters = response.json().get("clusters", [])
        if not clusters:
            raise Exception("No clusters available")
        
        return clusters[0]["cluster_id"]
    except Exception as e:
        print(f"Error getting cluster: {e}")
        return None

# Submit the job
def submit_job(workspace_path, cluster_id):
    try:
        # Create the job
        url = f"{DATABRICKS_HOST}/api/2.0/jobs/create"
        payload = {
            "name": f"{PREFIX}_churn_training_job",
            "tasks": [
                {
                    "task_key": "create_training_dataset",
                    "spark_python_task": {
                        "python_file": workspace_path,
                        "parameters": [f"--schema={SCHEMA_NAME}"]
                    },
                    "existing_cluster_id": cluster_id
                }
            ],
            "email_notifications": {
                "on_success": [],
                "on_failure": []
            },
            "timeout_seconds": 3600
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            raise Exception(f"Failed to create job: {response.text}")
        
        job_id = response.json()["job_id"]
        print(f"Created job: {job_id}")
        
        # Run the job
        url = f"{DATABRICKS_HOST}/api/2.0/jobs/run-now"
        payload = {
            "job_id": job_id
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            raise Exception(f"Failed to run job: {response.text}")
        
        run_id = response.json()["run_id"]
        print(f"Started job run: {run_id}")
        
        # Wait for completion
        url = f"{DATABRICKS_HOST}/api/2.0/jobs/runs/get"
        payload = {
            "run_id": run_id
        }
        
        while True:
            response = requests.get(url, headers=headers, params=payload)
            if response.status_code != 200:
                raise Exception(f"Failed to get run status: {response.text}")
            
            run_state = response.json().get("state", {})
            if run_state.get("life_cycle_state") in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
                break
        
        if run_state.get("result_state") == "SUCCESS":
            print("Job completed successfully.")
        else:
            print(f"Job failed: {run_state}")
        
    except Exception as e:
        print(f"Error submitting job: {e}")

if __name__ == "__main__":
    workspace_path = upload_job_script()
    if workspace_path:
        cluster_id = get_cluster_id()
        if cluster_id:
            submit_job(workspace_path, cluster_id)