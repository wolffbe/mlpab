#!/usr/bin/env python3

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, workspace, compute

# Initialize Databricks WorkspaceClient
w = WorkspaceClient()

# Define schema and job name
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")
job_name = f"{os.getenv('MLPAB_DATABRICKS_PREFIX')}_feature_table_job"
notebook_path = f"/Users/{w.current_user.me().user_name}/{os.getenv('MLPAB_DATABRICKS_PREFIX')}/transform_and_write"

# Create the notebook content
notebook_content = """
# Databricks notebook source
import pandas as pd
import numpy as np
import os

# Read input files from DBFS
requests_df = pd.read_csv("/dbfs/FileStore/requests.csv")
profiles_df = pd.read_csv("/dbfs/FileStore/profiles.csv")

# Merge requests and profiles on account_id
merged_df = pd.merge(requests_df, profiles_df, on="account_id", how="left")

# Compute distance_deg
def calculate_distance_deg(row):
    lat_diff = row["request_lat"] - row["home_lat"]
    lon_diff = row["request_lon"] - row["home_lon"]
    distance = np.sqrt(lat_diff**2 + lon_diff**2)
    return round(distance, 6)

merged_df["distance_deg"] = merged_df.apply(calculate_distance_deg, axis=1)

# Compute score
merged_df["score"] = merged_df.apply(
    lambda row: round(row["base_score"] - 0.1 * row["distance_deg"], 6), axis=1
)

# Select required columns
result_df = merged_df[["request_id", "account_id", "distance_deg", "score"]]

# Write to Delta table
table_name = f"{os.getenv('MLPAB_DATABRICKS_SCHEMA')}.scoreda4f6e2"
spark.createDataFrame(result_df).write.saveAsTable(
    name=table_name,
    mode="overwrite",
    format="delta"
)

# Enable online table for low-latency lookup
online_table_name = f"{os.getenv('MLPAB_DATABRICKS_SCHEMA')}.scoreda4f6e2_online"
spark.sql(f"CREATE OR REFRESH LIVE TABLE {online_table_name} AS SELECT * FROM {table_name}")

displayHTML("Feature table and online table created successfully.")
"""

# Create the parent directory
w.workspace.mkdirs(path=f"/Users/{w.current_user.me().user_name}/{os.getenv('MLPAB_DATABRICKS_PREFIX')}")

# Create the notebook in the workspace
w.workspace.upload(
    path=notebook_path + ".py",
    content=notebook_content.encode("utf-8"),
    format=workspace.ImportFormat.SOURCE,
    overwrite=True
)

# Define the job
job = w.jobs.create(
    name=job_name,
    tasks=[
        jobs.Task(
            task_key="transform_and_write",
            new_cluster=jobs.ClusterSpec(
                new_cluster=compute.ClusterSpec(
                    spark_version="13.3.x-scala2.12",
                    node_type_id="i3.xlarge",
                    num_workers=1,
                    autoscale=compute.AutoScale(min_workers=1, max_workers=2)
                )
            ),
            notebook_task=jobs.NotebookTask(
                notebook_path=notebook_path
            ),
            libraries=[
                compute.Library(pypi=compute.PythonPyPiLibrary(package="pandas")),
                compute.Library(pypi=compute.PythonPyPiLibrary(package="numpy"))
            ]
        )
    ]
)

# Run the job
run = w.jobs.run_now(job_id=job.job_id)

print(f"Job {job_name} submitted with run ID: {run.run_id}")
print(f"Notebook created at: {notebook_path}")