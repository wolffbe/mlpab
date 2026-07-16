#!/usr/bin/env python3
"""
Register a feature table named `incrementalaccaa3` (version 1) in the schema specified by MLPAB_DATABRICKS_SCHEMA.
Load all provided increments (data/increment_*.csv) into it.
Set up a recurring job named `incrementaljobaccaa3` to ingest future increments daily.
Enable online/real-time access for low-latency lookup.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, jobs, sql
from databricks.sdk.service.catalog import TableType, ColumnTypeName, ColumnInfo
from databricks.sdk.service.jobs import Task, NotebookTask, JobEmailNotifications, JobSettings, CronSchedule, PauseStatus

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.<run-id>
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab61b2a1
FEATURE_TABLE_NAME = "incrementalaccaa3"
JOB_NAME = f"{PREFIX}_incrementaljobaccaa3"
VOLUME_NAME = f"{PREFIX}_volume"
CATALOG, SCHEMA_NAME = SCHEMA.split(".")

# Initialize WorkspaceClient
w = WorkspaceClient()

# Create schema if it doesn't exist
try:
    w.schemas.get(f"{CATALOG}.{SCHEMA_NAME}")
except Exception:
    w.schemas.create(name=SCHEMA_NAME, catalog_name=CATALOG)

# Create a volume to stage the increment files
try:
    w.volumes.get(f"{CATALOG}.{SCHEMA_NAME}.{VOLUME_NAME}")
except Exception:
    w.volumes.create(
        name=VOLUME_NAME,
        catalog_name=CATALOG,
        schema_name=SCHEMA_NAME,
        volume_type=catalog.VolumeType.MANAGED
    )

# Upload increment files to the volume
volume_path = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}"
for i in range(1, 7):
    file_name = f"increment_{i:02d}.csv"
    local_path = f"data/{file_name}"
    with open(local_path, "rb") as f:
        w.files.upload(f"{volume_path}/{file_name}", f, overwrite=True)

# Create a table from the increment files
table_full_name = f"{CATALOG}.{SCHEMA_NAME}.{FEATURE_TABLE_NAME}"

# Define the schema for the table
columns = [
    ColumnInfo(name="row_id", type_name=ColumnTypeName.STRING, position=0, comment="unique record key"),
    ColumnInfo(name="account_id", type_name=ColumnTypeName.STRING, position=1),
    ColumnInfo(name="event_time", type_name=ColumnTypeName.BIGINT, position=2, comment="epoch milliseconds"),
    ColumnInfo(name="amount", type_name=ColumnTypeName.DOUBLE, position=3),
    ColumnInfo(name="category", type_name=ColumnTypeName.STRING, position=4),
]

# Create the table
try:
    w.tables.get(table_full_name)
except Exception:
    w.tables.create(
        name=FEATURE_TABLE_NAME,
        catalog_name=CATALOG,
        schema_name=SCHEMA_NAME,
        table_type=TableType.MANAGED,
        data_source_format=catalog.DataSourceFormat.DELTA,
        columns=columns,
        comment="Feature table for incremental ingestion of events data."
    )

# Load data from the volume into the table
for i in range(1, 7):
    file_name = f"increment_{i:02d}.csv"
    w.statement_execution.execute_statement(
        warehouse_id=w.warehouses.list()[0].id,
        catalog=CATALOG,
        schema=SCHEMA_NAME,
        statement=f"""
        COPY INTO {table_full_name}
        FROM '{volume_path}/{file_name}'
        FILEFORMAT = CSV
        FORMAT_OPTIONS('header' = 'true', 'inferSchema' = 'true')
        """
    ).result()

# Register the feature table
try:
    w.table_features.get(table_full_name)
except Exception:
    w.table_features.create(
        name=table_full_name,
        primary_keys=["row_id"],
        timestamp_keys=["event_time"],
        description="Feature table for incremental ingestion of events data."
    )

# Enable online table for low-latency lookup
try:
    w.online_tables.get(table_full_name)
except Exception:
    w.online_tables.create(
        name=table_full_name,
        spec=catalog.OnlineTableSpec(
            source_table_full_name=table_full_name,
            run_trigger=catalog.OnlineTableSpecTriggerType.AUTOMATIC
        )
    )

# Create a notebook for the ingestion job
notebook_path = f"/Users/{w.current_user.me().user_name}/{PREFIX}/incremental_ingest"
notebook_content = f"""
# Databricks notebook source
# MAGIC %md
# MAGIC ## Incremental Ingestion Job
# MAGIC 
# MAGIC This notebook ingests new increment files into the feature table `{table_full_name}`.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
import os

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.<run-id>
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab61b2a1
FEATURE_TABLE_NAME = "incrementalaccaa3"
VOLUME_NAME = f"{{PREFIX}}_volume"
CATALOG, SCHEMA_NAME = SCHEMA.split(".")

# Initialize WorkspaceClient
w = WorkspaceClient()

# Volume path
volume_path = f"/Volumes/{{CATALOG}}/{{SCHEMA_NAME}}/{{VOLUME_NAME}}"

# List new increment files in the volume
dbutils.fs.ls(volume_path)
files = [f.name for f in dbutils.fs.ls(volume_path) if f.name.startswith("increment_") and f.name.endswith(".csv")]

# Load new increments into the feature table
table_full_name = f"{CATALOG}.{SCHEMA_NAME}.{FEATURE_TABLE_NAME}"
for file_name in files:
    print(f"Loading {file_name} into {table_full_name}")
    spark.sql(f"COPY INTO {table_full_name} FROM '{volume_path}/{file_name}' FILEFORMAT = CSV FORMAT_OPTIONS('header' = 'true', 'inferSchema' = 'true')")

# COMMAND ----------

print("Ingestion complete.")
"""

# Create the notebook
try:
    w.workspace.get_status(notebook_path)
except Exception:
    w.workspace.upload(notebook_path, notebook_content.encode("utf-8"), overwrite=True)

# Create the recurring job
try:
    w.jobs.get_by_name(JOB_NAME)
except Exception:
    w.jobs.create(
        name=JOB_NAME,
        tasks=[
            Task(
                task_key="incremental_ingest",
                notebook_task=NotebookTask(
                    notebook_path=notebook_path,
                    base_parameters={{
                        "MLPAB_DATABRICKS_SCHEMA": SCHEMA,
                        "MLPAB_DATABRICKS_PREFIX": PREFIX
                    }}
                ),
                existing_cluster_id=w.clusters.list()[0].cluster_id
            )
        ],
        schedule=CronSchedule(
            quartz_cron_expression="0 0 0 * * ?",  # Daily at midnight
            timezone_id="UTC",
            pause_status=PauseStatus.UNPAUSED
        ),
        email_notifications=JobEmailNotifications(
            on_success=[w.current_user.me().user_name],
            on_failure=[w.current_user.me().user_name]
        )
    )

# Write submission/answers.json
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    f.write('{"job_name": "' + JOB_NAME + '"}')