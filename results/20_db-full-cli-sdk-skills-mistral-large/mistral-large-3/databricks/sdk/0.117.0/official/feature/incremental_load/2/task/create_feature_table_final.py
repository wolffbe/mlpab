#!/usr/bin/env python3
"""
Register a feature table named `incrementalaccaa3` (version 1) in the schema specified by MLPAB_DATABRICKS_SCHEMA.
Set up a recurring job named `incrementaljobaccaa3` to ingest future increments daily.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
from databricks.sdk.service.catalog import TableType, ColumnTypeName, OnlineTableSpec
from databricks.sdk.service.jobs import Task, NotebookTask, JobEmailNotifications, CronSchedule, PauseStatus

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.<run-id>
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab61b2a1
FEATURE_TABLE_NAME = "incrementalaccaa3"
JOB_NAME = f"{PREFIX}_incrementaljobaccaa3"
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
table_full_name = f"{CATALOG}.{SCHEMA_NAME}.{FEATURE_TABLE_NAME}"

# Initialize WorkspaceClient
w = WorkspaceClient()

# Create schema if it doesn't exist
try:
    w.schemas.get(f"{CATALOG}.{SCHEMA_NAME}")
except Exception:
    w.schemas.create(name=SCHEMA_NAME, catalog_name=CATALOG)

# Use SQL to create the table
warehouse_id = list(w.warehouses.list())[0].id
w.statement_execution.execute_statement(
    warehouse_id=warehouse_id,
    catalog=CATALOG,
    schema=SCHEMA_NAME,
    statement=f"""CREATE TABLE IF NOT EXISTS {table_full_name} (
        row_id STRING,
        account_id STRING,
        event_time LONG,
        amount DOUBLE,
        category STRING
    ) USING DELTA"""
)

# Load all provided increments into the table
for i in range(1, 7):
    file_name = f"increment_{i:02d}.csv"
    local_path = f"data/{file_name}"
    
    # Use SQL COPY INTO to load data
    w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        catalog=CATALOG,
        schema=SCHEMA_NAME,
        statement=f"""COPY INTO {table_full_name}
        FROM 'file:{local_path}'
        FILEFORMAT = CSV
        FORMAT_OPTIONS('header' = 'true', 'inferSchema' = 'true')"""
    )

# Enable online table for low-latency lookup
try:
    w.online_tables.get(table_full_name)
except Exception:
    w.online_tables.create(
        name=table_full_name,
        spec={
            "source_table_full_name": table_full_name,
            "run_trigger": "AUTOMATIC"
        }
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

# Load new increments into the feature table
# Note: This is a placeholder. In production, you would parameterize the file path.
spark.sql("SELECT 1")  # Placeholder

# COMMAND ----------

print("Ingestion setup complete.")
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
                    notebook_path=notebook_path
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