#!/usr/bin/env python3
"""
Creates a Databricks notebook to generate the training dataset and executes it.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace, jobs

# Environment variables
SCHEMA_NAME = os.environ["MLPAB_DATABRICKS_SCHEMA"].replace(".", "_")
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
NOTEBOOK_PATH = "/Users/{}/{}/create_training_dataset".format(os.environ.get("DATABRICKS_USER", "unknown_user"), PREFIX)

# Initialize WorkspaceClient
w = WorkspaceClient()

# Create a notebook to generate the training dataset
def create_notebook():
    notebook_content = f"""# Databricks notebook source
# MAGIC %md
# MAGIC # Create Training Dataset: churntrainingaf8b21_v1

# COMMAND ----------

# Create schema if not exists
spark.sql(\"CREATE SCHEMA IF NOT EXISTS workspace.{SCHEMA_NAME}\")

# COMMAND ----------

# Read CSV files directly from local path (assuming they are available)
transactions_df = spark.read.csv(\"/dbfs{os.path.abspath('./data/transactions.csv')}\", header=True, inferSchema=True)
transactions_late_df = spark.read.csv(\"/dbfs{os.path.abspath('./data/transactions_late.csv')}\", header=True, inferSchema=True)
profiles_df = spark.read.csv(\"/dbfs{os.path.abspath('./data/profiles.csv')}\", header=True, inferSchema=True)
activity_df = spark.read.csv(\"/dbfs{os.path.abspath('./data/activity.csv')}\", header=True, inferSchema=True)
account_health_df = spark.read.csv(\"/dbfs{os.path.abspath('./data/account_health.csv')}\", header=True, inferSchema=True)
labels_df = spark.read.csv(\"/dbfs{os.path.abspath('./data/labels.csv')}\", header=True, inferSchema=True)

# Register DataFrames as temporary views
transactions_df.createOrReplaceTempView(\"transactions\")
transactions_late_df.createOrReplaceTempView(\"transactions_late\")
profiles_df.createOrReplaceTempView(\"profiles\")
activity_df.createOrReplaceTempView(\"activity\")
account_health_df.createOrReplaceTempView(\"account_health\")
labels_df.createOrReplaceTempView(\"labels\")

# COMMAND ----------

# Create the training dataset by joining tables
spark.sql(f\"\"\"CREATE OR REPLACE TABLE workspace.{SCHEMA_NAME}.churntrainingaf8b21_v1 AS
WITH latest_transactions AS (
    SELECT 
        t.account_id,
        t.amount,
        t.balance,
        l.label_time
    FROM labels l
    LEFT JOIN (
        SELECT 
            account_id,
            amount,
            balance,
            event_time
        FROM (
            SELECT 
                account_id,
                amount,
                balance,
                event_time,
                ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
            FROM (
                SELECT * FROM transactions
                UNION ALL
                SELECT * FROM transactions_late
            )
        )
        WHERE rn = 1
    ) t
    ON l.account_id = t.account_id AND t.event_time <= l.label_time
),
latest_profiles AS (
    SELECT 
        p.account_id,
        p.credit_score,
        p.tier,
        l.label_time
    FROM labels l
    LEFT JOIN (
        SELECT 
            account_id,
            credit_score,
            tier,
            event_time
        FROM (
            SELECT 
                account_id,
                credit_score,
                tier,
                event_time,
                ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
            FROM profiles
        )
        WHERE rn = 1
    ) p
    ON l.account_id = p.account_id AND p.event_time <= l.label_time
),
latest_activity AS (
    SELECT 
        a.account_id,
        a.sessions_7d,
        l.label_time
    FROM labels l
    LEFT JOIN (
        SELECT 
            account_id,
            sessions_7d,
            event_time
        FROM (
            SELECT 
                account_id,
                sessions_7d,
                event_time,
                ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
            FROM activity
        )
        WHERE rn = 1
    ) a
    ON l.account_id = a.account_id AND a.event_time <= l.label_time
),
latest_health AS (
    SELECT 
        h.account_id,
        h.health_score,
        l.label_time
    FROM labels l
    LEFT JOIN (
        SELECT 
            account_id,
            health_score,
            event_time
        FROM (
            SELECT 
                account_id,
                health_score,
                event_time,
                ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
            FROM account_health
        )
        WHERE rn = 1
    ) h
    ON l.account_id = h.account_id AND h.event_time <= l.label_time
)
SELECT 
    l.account_id,
    l.label_time,
    t.amount,
    t.balance,
    p.credit_score,
    p.tier,
    a.sessions_7d,
    h.health_score,
    l.churned
FROM labels l
LEFT JOIN latest_transactions t ON l.account_id = t.account_id AND l.label_time = t.label_time
LEFT JOIN latest_profiles p ON l.account_id = p.account_id AND l.label_time = p.label_time
LEFT JOIN latest_activity a ON l.account_id = a.account_id AND l.label_time = a.label_time
LEFT JOIN latest_health h ON l.account_id = h.account_id AND l.label_time = h.label_time
\"\"\")

# COMMAND ----------

# Register the dataset as a versioned model
mlflow.set_registry_uri(\"databricks\")
model_uri = f\"runs:/dummy_run_id/workspace.{SCHEMA_NAME}.churntrainingaf8b21_v1\"
mlflow.register_model(model_uri, \"workspace.{SCHEMA_NAME}.churntrainingaf8b21\")

# COMMAND ----------

dbutils.notebook.exit(\"SUCCESS\")
"""

    # Create the notebook
    try:
        w.workspace.mkdirs(os.path.dirname(NOTEBOOK_PATH))
        w.workspace.upload(NOTEBOOK_PATH, notebook_content.encode("utf-8"), format=workspace.ImportFormat.SOURCE, overwrite=True)
        print(f"Created notebook: {NOTEBOOK_PATH}")
        return NOTEBOOK_PATH
    except Exception as e:
        print(f"Error creating notebook: {e}")
        return None

# Execute the notebook
def execute_notebook(notebook_path):
    try:
        # Create a job to run the notebook
        job = w.jobs.create(
            name=f"{PREFIX}_create_training_dataset",
            tasks=[
                jobs.NotebookTask(
                    notebook_path=notebook_path,
                    base_parameters={}
                )
            ],
            existing_cluster_id=w.clusters.list()[0].cluster_id  # Use the first available cluster
        )
        
        # Run the job
        run = w.jobs.run_now(job_id=job.job_id)
        print(f"Started job run: {run.run_id}")
        
        # Wait for completion
        run_result = w.jobs.wait_get_run_job_terminated_or_skipped(run_id=run.run_id)
        
        if run_result.result_state == jobs.RunResultState.SUCCESS:
            print("Training dataset created and registered successfully.")
        else:
            print(f"Job failed: {run_result.state}")
        
    except Exception as e:
        print(f"Error executing notebook: {e}")

if __name__ == "__main__":
    notebook_path = create_notebook()
    if notebook_path:
        execute_notebook(notebook_path)