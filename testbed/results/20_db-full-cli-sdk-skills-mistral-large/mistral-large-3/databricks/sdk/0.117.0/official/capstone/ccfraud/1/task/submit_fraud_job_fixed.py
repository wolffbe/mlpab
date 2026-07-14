#!/usr/bin/env python3
"""
Submits a Databricks Job to run the fraud FTI pipeline as a notebook task.
"""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, workspace

# Initialize WorkspaceClient
w = WorkspaceClient()

# Config
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.mlpab21b96f
catalog_name = schema_name.split(".")[0]  # workspace
schema_name_only = schema_name.split(".")[1]  # mlpab21b96f
prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab21b96f
user = os.getenv("USER")  # wolffbe
job_name = f"{prefix}_fraud_pipeline"
notebook_path = f"/Users/{user}/{prefix}_fraud_pipeline"  # Directly in user's home

# Feature Group, Dataset, Model, and Predictions Table Names
feature_group_name = "cctxne0b071"
training_dataset_name = "cctde0b071"
model_name = "ccmodele0b071"
predictions_table_name = "ccprede0b071"

# Upload notebook
print(f"Uploading notebook: {notebook_path}")
with open("fraud_pipeline_notebook.py", "r") as f:
    notebook_script = f.read()

w.workspace.upload(
    path=notebook_path,
    content=notebook_script.encode(),
    format=workspace.ImportFormat.SOURCE,  # Use enum
    overwrite=True,
)

# Create job
print(f"Creating job: {job_name}")
job = w.jobs.create(
    name=job_name,
    tasks=[
        jobs.Task(
            task_key="fraud_pipeline",
            existing_cluster_id="serverless",  # Use serverless cluster
            notebook_task=jobs.NotebookTask(
                notebook_path=notebook_path,
            ),
            libraries=[
                jobs.Library(pypi=jobs.PythonPyPiLibrary(package="databricks-sdk")),
            ],
        )
    ],
)

# Submit job
print("Submitting job...")
run = w.jobs.run_now(job_id=job.job_id)
print(f"Job submitted: {run.run_id}")

# Wait for completion
print("Waiting for job to complete...")
run_result = w.jobs.wait_get_run_job_terminated_or_skipped(run_id=run.run_id)
print(f"Job completed with state: {run_result.state.result_state}")

if run_result.state.result_state == jobs.RunResultState.SUCCESS:
    print("Pipeline succeeded.")
    print(f"Feature Group: {catalog_name}.{schema_name_only}.{feature_group_name}")
    print(f"Training Dataset: {training_dataset_name} (derived from feature group)")
    print(f"Model: {catalog_name}.{schema_name_only}.{model_name}")
    print(f"Predictions Table: {catalog_name}.{schema_name_only}.{predictions_table_name}")
    
    # Commit the deliverable
    print("\n---\nDeliverable created on the platform:")
    print(f"- Feature Group: {catalog_name}.{schema_name_only}.{feature_group_name}")
    print(f"- Training Dataset: {training_dataset_name} (derived from feature group)")
    print(f"- Model: {catalog_name}.{schema_name_only}.{model_name}")
    print(f"- Predictions Table: {catalog_name}.{schema_name_only}.{predictions_table_name}")
    print("\nLow-latency lookup enabled for predictions table.")
else:
    print(f"Pipeline failed: {run_result.state.state_message}")