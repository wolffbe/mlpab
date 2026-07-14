#!/usr/bin/env python3
"""
Run a one-time job to create the feature table from a CSV file in a Unity Catalog volume.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

# Initialize Databricks WorkspaceClient
w = WorkspaceClient()

# Get schema and prefix from environment variables
full_schema_name = os.environ['MLPAB_DATABRICKS_SCHEMA']
schema_name = full_schema_name.split(".")[-1]
table_name = "recse3a36e"
full_table_name = f"{full_schema_name}.{table_name}"
volume_name = f"{os.environ['MLPAB_DATABRICKS_PREFIX']}_volume"
full_volume_name = f"{full_schema_name}.{volume_name}"

# Path to the recommendations CSV in the volume
volume_path = f"/Volumes/workspace/{schema_name}/{volume_name}/recommendations.csv"

# Path to the script in the volume
script_path = f"/Volumes/workspace/{schema_name}/{volume_name}/create_table_script.py"

# Define the job to run the script
job_name = f"{os.environ['MLPAB_DATABRICKS_PREFIX']}_create_table"
job_settings = {
    "run_name": job_name,
    "cluster_spec": {
        "new_cluster": {
            "spark_version": "14.3.x-scala2.12",
            "node_type_id": "i3.xlarge",
            "num_workers": 1
        }
    },
    "spark_python_task": {
        "python_file": f"dbfs:{script_path}"
    }
}

# Submit the job
run = w.jobs.submit(
    run_name=job_name,
    new_cluster={
        "spark_version": "14.3.x-scala2.12",
        "node_type_id": "i3.xlarge",
        "num_workers": 1
    },
    spark_python_task={
        "python_file": f"dbfs:{script_path}"
    }
).result()
print(f"Job {job_name} completed with run_id {run.run_id}.")

print(f"Feature table {full_table_name} is ready for online access.")