# Databricks notebook source
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
import os

w = WorkspaceClient()

catalog = "workspace"
schema = "mlpabf104af"

# Create the table
w.statement_execution.execute_statement(
    warehouse_id="4dfab06c923fe3cc",
    catalog=catalog,
    schema=schema,
    statement="CREATE TABLE IF NOT EXISTS profiles395e7c (account_id STRING, f1 DOUBLE, f2 DOUBLE, f3 DOUBLE, f4 DOUBLE);"
)

# Load the data
w.statement_execution.execute_statement(
    warehouse_id="4dfab06c923fe3cc",
    catalog=catalog,
    schema=schema,
    statement=f"""
    INSERT INTO profiles395e7c
    SELECT * FROM csv."/Users/benedict@logicalclocks.com/{os.environ['MLPAB_DATABRICKS_PREFIX']}/features.csv"
    (header => true, inferSchema => true);
    """
)