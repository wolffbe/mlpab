#!/usr/bin/env python3
"""
Submits a Delta Live Table (DLT) Pipeline for the fraud FTI pipeline.
"""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import pipelines

# Initialize WorkspaceClient
w = WorkspaceClient()

# Config
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.mlpab21b96f
catalog_name = schema_name.split(".")[0]  # workspace
schema_name_only = schema_name.split(".")[1]  # mlpab21b96f
prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab21b96f
user = os.getenv("USER")  # wolffbe
pipeline_name = f"{prefix}_fraud_pipeline"

# Feature Group, Dataset, Model, and Predictions Table Names
feature_group_name = "cctxne0b071"
training_dataset_name = "cctde0b071"
model_name = "ccmodele0b071"
predictions_table_name = "ccprede0b071"

# Upload DLT pipeline script to DBFS
print(f"Uploading DLT pipeline script to DBFS: /Users/{user}/{prefix}/dlt_fraud_pipeline.py")
with open("dlt_fraud_pipeline.py", "r") as f:
    pipeline_script = f.read()

w.dbfs.mkdirs(f"dbfs:/Users/{user}/{prefix}/")
w.dbfs.upload(
    f"dbfs:/Users/{user}/{prefix}/dlt_fraud_pipeline.py",
    pipeline_script.encode(),
    overwrite=True,
)

# Create DLT Pipeline
print(f"Creating DLT pipeline: {pipeline_name}")
pipeline = w.pipelines.create(
    name=pipeline_name,
    storage=f"/Users/{user}/{prefix}/pipeline_storage",
    configuration={
        "spark.master": "local[*]",
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
                path=f"/Users/{user}/{prefix}/dlt_fraud_pipeline.py",
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
else:
    print(f"Pipeline failed: {run_result.state}")