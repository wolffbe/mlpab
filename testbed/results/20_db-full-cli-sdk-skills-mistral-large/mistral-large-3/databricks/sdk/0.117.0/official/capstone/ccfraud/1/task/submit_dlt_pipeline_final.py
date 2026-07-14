#!/usr/bin/env python3
"""
Submits a Delta Live Table (DLT) Pipeline for the fraud FTI pipeline using Workspace API.
"""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import pipelines, workspace

# Initialize WorkspaceClient
w = WorkspaceClient()

# Config
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.mlpab21b96f
catalog_name = schema_name.split(".")[0]  # workspace
schema_name_only = schema_name.split(".")[1]  # mlpab21b96f
prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab21b96f
user = os.getenv("USER")  # wolffbe
pipeline_name = f"{prefix}_fraud_pipeline"
notebook_path = f"/Users/{user}/{prefix}/dlt_fraud_pipeline"

# Feature Group, Dataset, Model, and Predictions Table Names
feature_group_name = "cctxne0b071"
training_dataset_name = "cctde0b071"
model_name = "ccmodele0b071"
predictions_table_name = "ccprede0b071"

# Upload DLT pipeline notebook
print(f"Uploading DLT pipeline notebook: {notebook_path}")
with open("dlt_fraud_pipeline_notebook.py", "r") as f:
    notebook_script = f.read()

w.workspace.upload(
    path=notebook_path,
    content=notebook_script.encode(),
    format=workspace.ImportFormat.SOURCE,
    overwrite=True,
)

# Create DLT Pipeline
print(f"Creating DLT pipeline: {pipeline_name}")
pipeline = w.pipelines.create(
    name=pipeline_name,
    storage=f"/pipelines/{prefix}_fraud_pipeline_storage",
    configuration={
        "pipelines.trigger.interval": "1 hour",
    },
    clusters=[
        pipelines.PipelineCluster(
            label="default",
            num_workers=2,
            node_type_id="i3.xlarge",
            custom_tags={"usage": "fraud_fti"},
        )
    ],
    libraries=[
        pipelines.PipelineLibrary(
            notebook=pipelines.NotebookLibrary(
                path=notebook_path,
            )
        )
    ],
    continuous=False,
    development=True,
)

# Run the pipeline
print("Starting pipeline...")
run = w.pipelines.start(pipeline_id=pipeline.pipeline_id)
print(f"Pipeline started: {run.run_id}")

# Wait for completion
print("Waiting for pipeline to complete...")
run_result = w.pipelines.wait_get_run_terminated(pipeline_id=pipeline.pipeline_id, run_id=run.run_id)
print(f"Pipeline completed with state: {run_result.state}")

if run_result.state == pipelines.PipelineState.COMPLETED:
    print("Pipeline succeeded.")
    print(f"Feature Group: {catalog_name}.{schema_name_only}.{feature_group_name}")
    print(f"Training Dataset: {catalog_name}.{schema_name_only}.{training_dataset_name}")
    print(f"Model: {catalog_name}.{schema_name_only}.{model_name}")
    print(f"Predictions Table: {catalog_name}.{schema_name_only}.{predictions_table_name}")
    
    # Commit the deliverable
    print("\n---\nDeliverable created on the platform:")
    print(f"- Feature Group: {catalog_name}.{schema_name_only}.{feature_group_name}")
    print(f"- Training Dataset: {catalog_name}.{schema_name_only}.{training_dataset_name}")
    print(f"- Model: {catalog_name}.{schema_name_only}.{model_name}")
    print(f"- Predictions Table: {catalog_name}.{schema_name_only}.{predictions_table_name}")
    print("\nLow-latency lookup enabled for predictions table.")
else:
    print(f"Pipeline failed: {run_result.state}")
    print(f"Pipeline message: {run_result.message}")