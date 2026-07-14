"""Script to create and run a Databricks job using the API directly."""
import os
import time
import requests
import json

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")
JOB_NAME = "trainjobac536a"
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST").strip()
if not DATABRICKS_HOST.startswith(('http://', 'https://')):
    DATABRICKS_HOST = f"https://{DATABRICKS_HOST}"
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# Job definition
job_definition = {
    "name": JOB_NAME,
    "tasks": [
        {
            "task_key": "train_task",
            "new_cluster": {
                "spark_version": "14.3.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 0,
                "spark_conf": {
                    "spark.databricks.cluster.profile": "singleNode",
                    "spark.master": "local[*]",
                },
                "custom_tags": {"resource_class": "single-node"},
            },
            "spark_python_task": {
                "python_file": "dbfs:/FileStore/train_model.py",
            },
            "libraries": [
                {
                    "pypi": {
                        "package": "pandas==2.0.3"
                    }
                },
                {
                    "pypi": {
                        "package": "numpy==1.24.3"
                    }
                },
            ],
            "timeout_seconds": 3600,
        }
    ],
    "max_concurrent_runs": 1,
}

# Upload the training script and data to DBFS
session = requests.Session()
session.headers.update({"Authorization": f"Bearer {DATABRICKS_TOKEN}"})

# Create DBFS directory
session.post(f"{DATABRICKS_HOST}/api/2.0/dbfs/mkdirs", json={"path": "/FileStore/"})

# Upload files
with open("./data/train_model.py", "rb") as f:
    session.post(
        f"{DATABRICKS_HOST}/api/2.0/dbfs/put",
        data={"path": "/FileStore/train_model.py", "overwrite": "true"},
        files={"file": f},
    )

with open("./data/train.csv", "rb") as f:
    session.post(
        f"{DATABRICKS_HOST}/api/2.0/dbfs/put",
        data={"path": "/FileStore/train.csv", "overwrite": "true"},
        files={"file": f},
    )

with open("./data/score.csv", "rb") as f:
    session.post(
        f"{DATABRICKS_HOST}/api/2.0/dbfs/put",
        data={"path": "/FileStore/score.csv", "overwrite": "true"},
        files={"file": f},
    )

# Create the job
response = session.post(
    f"{DATABRICKS_HOST}/api/2.1/jobs/create",
    json=job_definition,
)
response.raise_for_status()
job_id = response.json()["job_id"]
print(f"Created job with ID: {job_id}")

# Start the job
response = session.post(
    f"{DATABRICKS_HOST}/api/2.1/jobs/run-now",
    json={"job_id": job_id},
)
response.raise_for_status()
run_id = response.json()["run_id"]
print(f"Started job run with ID: {run_id}")

# Wait for the job to complete
while True:
    response = session.get(
        f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get",
        params={"run_id": run_id},
    )
    response.raise_for_status()
    run_status = response.json()
    
    if run_status["state"]["life_cycle_state"] == "TERMINATED":
        if run_status["state"]["result_state"] == "SUCCESS":
            print("Job completed successfully.")
            break
        else:
            raise Exception(f"Job failed with state: {run_status['state']['result_state']}")
    time.sleep(10)

# Download the predictions file
with open("./predictions.csv", "wb") as f:
    response = session.get(
        f"{DATABRICKS_HOST}/api/2.0/dbfs/read",
        params={"path": "/FileStore/predictions.csv"},
    )
    response.raise_for_status()
    f.write(response.content)

print("Downloaded predictions.csv")