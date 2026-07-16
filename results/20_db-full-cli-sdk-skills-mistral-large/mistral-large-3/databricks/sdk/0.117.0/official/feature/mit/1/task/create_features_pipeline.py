#!/usr/bin/env python3
"""
Creates and runs a Delta Live Tables (DLT) pipeline to compute the required features
from transactions.csv and fx_rates.csv, then enables online access for the table.
"""

import os
import time
import io
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import pipelines, catalog, workspace

# Environment variables
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # catalog.schema
CATALOG, SCHEMA_NAME = SCHEMA.split(".")

# Pipeline and table names
PIPELINE_NAME = f"{PREFIX}_featuresfbc05f_pipeline"
TABLE_NAME = "featuresfbc05f"
FULL_TABLE_NAME = f"{SCHEMA}.{TABLE_NAME}"

# Input paths (in Unity Catalog Volume)
VOLUME_NAME = "feature_data"
TRANSACTIONS_PATH = f"{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}/transactions.csv"
FX_RATES_PATH = f"{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}/fx_rates.csv"

# DLT Pipeline definition (as a notebook)
PIPELINE_NOTEBOOK_CONTENT = f"""
# Databricks notebook source
import dlt
from pyspark.sql.functions import col, expr, when
from pyspark.sql.window import Window

@dlt.table(
    name="{TABLE_NAME}",
    comment="Feature table for transactions with USD conversion, weekend flag, and 7-day rolling sum.",
    table_properties={{
        "quality": "gold",
        "delta.autoOptimize.optimizeWrite": "true",
        "delta.autoOptimize.autoCompact": "true"
    }}
)
@dlt.expect_or_drop("valid_row_id", "row_id IS NOT NULL")
@dlt.expect_or_drop("valid_account_id", "account_id IS NOT NULL")
@dlt.expect_or_drop("valid_event_time", "event_time IS NOT NULL")
@dlt.expect_or_drop("valid_amount", "amount IS NOT NULL")
@dlt.expect_or_drop("valid_currency", "currency IS NOT NULL")
def {TABLE_NAME}():
    # Read raw data
    transactions = dlt.read("transactions").alias("t")
    fx_rates = dlt.read("fx_rates").alias("fx")
    
    # Join and compute features
    joined = transactions.join(
        fx_rates, 
        transactions.currency == fx_rates.currency,
        "inner"
    ).select(
        col("t.row_id"),
        col("t.account_id"),
        col("t.event_time"),
        (col("t.amount") * col("fx.fx_rate")).alias("amount_usd"),
        col("t.currency"),
        col("t.amount")
    )
    
    # Weekend flag (UTC)
    result = joined.withColumn(
        "is_weekend",
        when(expr("date_format(to_timestamp(event_time / 1000), 'E') IN ('Sat', 'Sun')"), 1).otherwise(0)
    )
    
    # 7-day rolling sum for each account
    window = Window.partitionBy("account_id").orderBy("event_time").rangeBetween(-7 * 24 * 60 * 60 * 1000, 0)
    result = result.withColumn("amount_7d", F.sum("amount").over(window))
    
    return result.select(
        "row_id",
        "account_id",
        "event_time",
        "amount_usd",
        "is_weekend",
        "amount_7d"
    )

@dlt.table(name="transactions")
def transactions():
    return spark.read.csv("{TRANSACTIONS_PATH}", header=True, inferSchema=True)

@dlt.table(name="fx_rates")
@dlt.expect_or_drop("valid_currency", "currency IS NOT NULL")
@dlt.expect_or_drop("valid_fx_rate", "fx_rate IS NOT NULL")
def fx_rates():
    return spark.read.csv("{FX_RATES_PATH}", header=True, inferSchema=True)
"""


def main():
    w = WorkspaceClient()
    
    # Assume files are already in the Volume (Volume already exists)
    print("Assuming input files are already in the Volume...")
    # Hardcode the Volume path in the DLT pipeline
    global TRANSACTIONS_PATH, FX_RATES_PATH
    TRANSACTIONS_PATH = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}/transactions.csv"
    FX_RATES_PATH = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}/fx_rates.csv"
    
    # Create the notebook for the DLT pipeline (use /Shared for accessibility)
    notebook_path = f"/Shared/{PREFIX}/dlt_pipeline_notebook"
    print(f"Creating notebook at {notebook_path}...")
    w.workspace.mkdirs(os.path.dirname(notebook_path))
    import base64
    w.workspace.import_(
        path=notebook_path,
        content=base64.b64encode(PIPELINE_NOTEBOOK_CONTENT.encode("utf-8")).decode("utf-8"),
        format=workspace.ImportFormat.SOURCE,
        overwrite=True,
        language=workspace.Language.PYTHON
    )
    
    # Create the DLT pipeline
    print(f"Creating DLT pipeline {PIPELINE_NAME}...")
    pipeline = w.pipelines.create(
        name=PIPELINE_NAME,
        configuration={
            "pipelines.cluster.type": "SERVERLESS",
            "pipelines.serverless.enabled": "true",
            "pipelines.serverless.compute.enable": "true"
        },

        libraries=[
            pipelines.PipelineLibrary(
                notebook=pipelines.NotebookLibrary(
                    path=notebook_path
                )
            )
        ],
        continuous=False,
        development=True,
        photon=True,
        catalog=CATALOG,
        target=SCHEMA_NAME,
        edition="ADVANCED",
        channel="CURRENT"
    )
    
    # Run the pipeline
    print(f"Starting pipeline {PIPELINE_NAME}...")
    run = w.pipelines.start(pipeline_id=pipeline.pipeline_id)
    print(f"Pipeline run ID: {run.run_id}")
    
    # Wait for completion (polling)
    run = w.pipelines.get_run(run_id=run.run_id)
    while run.state in (pipelines.PipelineState.RUNNING, pipelines.PipelineState.STARTING, pipelines.PipelineState.PENDING):
        run = w.pipelines.get_run(run_id=run.run_id)
        print(f"Pipeline state: {run.state}")
        time.sleep(10)
    
    if run.state != pipelines.PipelineState.COMPLETED:
        raise Exception(f"Pipeline failed with state: {run.state}")
    
    print(f"Pipeline completed successfully. Table {FULL_TABLE_NAME} is ready.")
    
    # Enable online access for the table
    print(f"Enabling online access for {FULL_TABLE_NAME}...")
    w.online_tables.create(
        name=FULL_TABLE_NAME,
        spec=pipelines.OnlineTableSpec(
            source_table_full_name=FULL_TABLE_NAME,
            run_triggered=True
        )
    )
    
    print(f"Online table for {FULL_TABLE_NAME} is now available for low-latency access.")


if __name__ == "__main__":
    main()