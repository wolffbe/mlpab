#!/usr/bin/env python3
"""
Filter events.csv to keep only rows that satisfy the data contract.
Write submission/answers.json with rejected row IDs.
Register a feature table named `events45bd4b` (version 1) in the schema `workspace.mlpab7c6cbb`
with record key `row_id` and event-time column `event_time`.
Enable online/real-time access for the table.
"""

import csv
import json
import os
import tempfile
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.catalog import *
from databricks.sdk.service.compute import *

# Data contract rules
VALID_CATEGORIES = {"grocery", "travel", "salary", "rent", "other"}


def is_valid_row(row):
    """Check if a row satisfies all contract rules."""
    try:
        amount = row['amount']
        if amount == '':
            return False
        amount = float(amount)
        if not (0 <= amount <= 10000):
            return False
        if row['category'] not in VALID_CATEGORIES:
            return False
        return True
    except (ValueError, KeyError):
        return False


def main():
    # Read and filter events
    valid_rows = []
    rejected_row_ids = []
    
    with open('data/events.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_valid_row(row):
                valid_rows.append(row)
            else:
                rejected_row_ids.append(row['row_id'])
    
    # Write rejected row IDs to submission/answers.json
    with open('submission/answers.json', 'w') as f:
        json.dump({"rejected": rejected_row_ids}, f, indent=2)
    
    # Initialize Databricks WorkspaceClient
    w = WorkspaceClient()
    
    # Schema details
    catalog_name = "workspace"
    schema_name = os.environ["MLPAB_DATABRICKS_SCHEMA"]
    table_name = "events45bd4b"
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
    
    # Prefix for job name
    job_prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
    job_name = f"{job_prefix}_filter_events"
    
    # Create a temporary file with the valid rows
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        fieldnames = ['row_id', 'account_id', 'event_time', 'amount', 'category']
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_rows)
        tmp_path = tmp.name
    
    # Upload the valid rows CSV to DBFS
    dbfs_path = f"/tmp/{job_prefix}_valid_events.csv"
    w.dbfs.upload(tmp_path, dbfs_path, overwrite=True)
    
    # Create a cluster if none exists (use smallest available)
    clusters = list(w.clusters.list())
    if not clusters:
        cluster = w.clusters.create(
            cluster_name=f"{job_prefix}_cluster",
            spark_version="14.3.x-scala2.12",
            node_type_id="i3.xlarge",
            autoscale=AutoScale(min_workers=1, max_workers=2),
            num_workers=1
        ).result()
    else:
        cluster = clusters[0]
    
    # Define the job task to register the feature table
    job_task = jobs.SubmitTask(
        task_key="register_feature_table",
        existing_cluster_id=cluster.cluster_id,
        spark_python_task=jobs.SparkPythonTask(
            python_file=f"dbfs:{dbfs_path}",
            parameters=[
                "--catalog", catalog_name,
                "--schema", schema_name,
                "--table", table_name,
                "--dbfs-path", dbfs_path
            ]
        ),
        libraries=[
            jobs.Library(pypi=jobs.PythonPyPiLibrary(package="databricks-sdk"))
        ]
    )
    
    # Submit the job
    job = w.jobs.submit(
        name=job_name,
        tasks=[job_task]
    ).result()
    
    # Wait for job to complete
    job_run = w.jobs.get_run(job.run_id)
    while job_run.state.life_cycle_state not in [jobs.RunLifeCycleState.TERMINATED, jobs.RunLifeCycleState.SKIPPED, jobs.RunLifeCycleState.INTERNAL_ERROR]:
        job_run = w.jobs.get_run(job.run_id)
    
    if job_run.state.result_state != jobs.RunResultState.SUCCESS:
        raise Exception(f"Job failed: {job_run.state.state_message}")
    
    # Register the feature table using the SDK
    w.tables.create(
        name=table_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_type=TableType.MANAGED,
        data_source_format=DataSourceFormat.DELTA,
        storage_location=f"dbfs:{dbfs_path.replace('.csv', '')}",
        table_constraints=[
            TableConstraint(
                primary_key_constraint=PrimaryKeyConstraint(name="row_id_pk", columns=["row_id"])
            )
        ],
        properties={
            "delta.enableChangeDataFeed": "true",
            "pipelines.autoOptimize.managed": "true"
        }
    )
    
    # Enable online table for real-time access
    w.online_tables.create(
        name=full_table_name,
        spec=OnlineTableSpec(
            source_table_full_name=full_table_name,
            primary_key_columns=["row_id"],
            timeseries_key="event_time"
        )
    )
    
    print("Feature table registered and online access enabled.")


if __name__ == "__main__":
    main()