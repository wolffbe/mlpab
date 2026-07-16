#!/usr/bin/env python3
"""
Create a feature table from a CSV file in a Unity Catalog volume.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, jobs

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

# Write the script to a local file
local_script_path = "create_table_script.py"

# Read the template script
with open("create_table_script.py", "r") as f:
    script_content = f.read()

# Replace placeholders
script_content = script_content.replace("FULL_TABLE_NAME", full_table_name)
script_content = script_content.replace("VOLUME_PATH", volume_path)

# Write the updated script
with open(local_script_path, "w") as f:
    f.write(script_content)

# Upload the script to the volume
script_path = f"/Volumes/workspace/{schema_name}/{volume_name}/create_table_script.py"

# Upload the script to the volume
with open(local_script_path, "rb") as f:
    w.files.upload(script_path, f, overwrite=True)
print(f"Uploaded script to {script_path}.")

# Define the job to run the script
job_name = f"{os.environ['MLPAB_DATABRICKS_PREFIX']}_create_table"
job_settings = {
    "name": job_name,
    "tasks": [
        {
            "task_key": "create_table",
            "new_cluster": {
                "spark_version": "14.3.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 1
            },
            "spark_python_task": {
                "python_file": f"dbfs:{script_path}"
            }
        }
    ]
}

# Create the job
job = w.jobs.create(**job_settings)
print(f"Job {job_name} created with job_id {job.job_id}.")

# Run the job
run = w.jobs.run_now(job_id=job.job_id).result()
print(f"Job {job_name} completed with run_id {run.run_id}.")

print(f"Feature table {full_table_name} is ready for online access.")