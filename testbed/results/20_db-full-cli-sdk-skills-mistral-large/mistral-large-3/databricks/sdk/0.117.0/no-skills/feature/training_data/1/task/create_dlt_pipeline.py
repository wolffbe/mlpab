#!/usr/bin/env python3
"""
Creates a Delta Live Table (DLT) pipeline to ingest CSV files and produce the training dataset.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import pipelines

# Environment variables
SCHEMA_NAME = os.environ["MLPAB_DATABRICKS_SCHEMA"].replace(".", "_")
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
DATA_DIR = os.path.abspath("./data")

# Initialize WorkspaceClient
w = WorkspaceClient()

# Create a DLT pipeline to ingest and transform data
def create_dlt_pipeline():
    try:
        # Define the pipeline
        pipeline = w.pipelines.create(
            name=f"{PREFIX}_churn_training_pipeline",
            storage=f"/pipelines/{PREFIX}_churn_training",
            configuration={
                "spark.sql.catalog.workspace": "com.databricks.spark.sql.catalog.delta.Catalog",
                "spark.sql.catalog.workspace.type": "hive_metastore",
                "spark.sql.catalog.workspace.database": SCHEMA_NAME,
                "input.data.dir": f"file:{DATA_DIR}"
            },
            clusters=[
                pipelines.PipelineCluster(
                    label="default",
                    num_workers=1,
                    node_type_id=w.clusters.list_node_types()[0].node_type_id,
                    spark_conf={
                        "spark.databricks.delta.preview.enabled": "true"
                    }
                )
            ],
            libraries=[
                pipelines.PipelineLibrary(
                    notebook=pipelines.NotebookLibrary(
                        path="/Shared/churn_training_dlt"  # Placeholder, will be created below
                    )
                )
            ],
            continuous=False,
            development=True,
            photon=True,
            edition="ADVANCED",
            channel="CURRENT"
        )
        
        print(f"Created DLT pipeline: {pipeline.name}")
        return pipeline.pipeline_id
    except Exception as e:
        print(f"Error creating DLT pipeline: {e}")
        return None

# Create the DLT notebook
def create_dlt_notebook():
    notebook_content = f"""# Databricks notebook source
# DLT Pipeline for Churn Training Dataset

import dlt
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window

# Read CSV files
@dlt.table(name="transactions_raw")
def get_transactions():
    return spark.read.csv(f"file:{DATA_DIR}/transactions.csv", header=True, inferSchema=True)

@dlt.table(name="transactions_late_raw")
def get_transactions_late():
    return spark.read.csv(f"file:{DATA_DIR}/transactions_late.csv", header=True, inferSchema=True)

@dlt.table(name="profiles_raw")
def get_profiles():
    return spark.read.csv(f"file:{DATA_DIR}/profiles.csv", header=True, inferSchema=True)

@dlt.table(name="activity_raw")
def get_activity():
    return spark.read.csv(f"file:{DATA_DIR}/activity.csv", header=True, inferSchema=True)

@dlt.table(name="account_health_raw")
def get_account_health():
    return spark.read.csv(f"file:{DATA_DIR}/account_health.csv", header=True, inferSchema=True)

@dlt.table(name="labels_raw")
def get_labels():
    return spark.read.csv(f"file:{DATA_DIR}/labels.csv", header=True, inferSchema=True)

# Create the training dataset
@dlt.table(name="churntrainingaf8b21_v1")
def create_training_dataset():
    # Get the most recent feature values at or before label_time
    labels_df = dlt.read("labels_raw")
    
    # Transactions (union of both tables)
    transactions_df = dlt.read("transactions_raw").union(dlt.read("transactions_late_raw"))
    window = Window.partitionBy("account_id").orderBy(col("event_time").desc())
    latest_transactions = transactions_df.withColumn("rn", row_number().over(window)).filter(col("rn") == 1).drop("rn")
    
    # Profiles
    profiles_df = dlt.read("profiles_raw")
    latest_profiles = profiles_df.withColumn("rn", row_number().over(window)).filter(col("rn") == 1).drop("rn")
    
    # Activity
    activity_df = dlt.read("activity_raw")
    latest_activity = activity_df.withColumn("rn", row_number().over(window)).filter(col("rn") == 1).drop("rn")
    
    # Account Health
    account_health_df = dlt.read("account_health_raw")
    latest_health = account_health_df.withColumn("rn", row_number().over(window)).filter(col("rn") == 1).drop("rn")
    
    # Join with labels to get the most recent features at or before label_time
    result = labels_df.alias("l") \
        .join(
            latest_transactions.alias("t"), 
            (col("l.account_id") == col("t.account_id")) & (col("t.event_time") <= col("l.label_time")), 
            "left"
        ) \
        .join(
            latest_profiles.alias("p"), 
            (col("l.account_id") == col("p.account_id")) & (col("p.event_time") <= col("l.label_time")), 
            "left"
        ) \
        .join(
            latest_activity.alias("a"), 
            (col("l.account_id") == col("a.account_id")) & (col("a.event_time") <= col("l.label_time")), 
            "left"
        ) \
        .join(
            latest_health.alias("h"), 
            (col("l.account_id") == col("h.account_id")) & (col("h.event_time") <= col("l.label_time")), 
            "left"
        ) \
        .select(
            col("l.account_id"),
            col("l.label_time"),
            col("t.amount"),
            col("t.balance"),
            col("p.credit_score"),
            col("p.tier"),
            col("a.sessions_7d"),
            col("h.health_score"),
            col("l.churned")
        )
    
    return result

# Register the dataset as a versioned model
@dlt.table(name="churntrainingaf8b21_model")
def register_model():
    import mlflow
    mlflow.set_registry_uri("databricks")
    model_uri = f"runs:/dummy_run_id/workspace.{SCHEMA_NAME}.churntrainingaf8b21_v1"
    mlflow.register_model(model_uri, f"workspace.{SCHEMA_NAME}.churntrainingaf8b21")
    return spark.createDataFrame([(1,)], ["dummy"])
"""

    # Upload the notebook to a shared location
    notebook_path = "/Shared/churn_training_dlt"
    try:
        w.workspace.upload(notebook_path, notebook_content.encode("utf-8"), format=workspace.ImportFormat.SOURCE, overwrite=True)
        print(f"Created DLT notebook: {notebook_path}")
        return notebook_path
    except Exception as e:
        print(f"Error creating DLT notebook: {e}")
        return None

# Run the DLT pipeline
def run_dlt_pipeline(pipeline_id):
    try:
        # Start the pipeline
        update = w.pipelines.start_update(pipeline_id=pipeline_id)
        print(f"Started DLT pipeline update: {update.update_id}")
        
        # Wait for completion
        result = w.pipelines.wait_get_update(pipeline_id=pipeline_id, update_id=update.update_id)
        
        if result.state == pipelines.PipelineState.COMPLETED:
            print("DLT pipeline completed successfully.")
        else:
            print(f"DLT pipeline failed: {result.state}")
        
    except Exception as e:
        print(f"Error running DLT pipeline: {e}")

if __name__ == "__main__":
    # Create the DLT notebook
    notebook_path = create_dlt_notebook()
    if notebook_path:
        # Create the DLT pipeline
        pipeline_id = create_dlt_pipeline()
        if pipeline_id:
            # Run the DLT pipeline
            run_dlt_pipeline(pipeline_id)