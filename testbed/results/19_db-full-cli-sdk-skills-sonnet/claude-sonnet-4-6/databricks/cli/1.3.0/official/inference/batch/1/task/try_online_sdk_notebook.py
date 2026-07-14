# Databricks notebook source
# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
)

w = WorkspaceClient()

try:
    result = w.online_tables.create(table=OnlineTable(
        name="workspace.mlpab6ef9cb.scores4f5893_online",
        spec=OnlineTableSpec(
            source_table_full_name="workspace.mlpab6ef9cb.scores4f5893",
            primary_key_columns=["account_id"],
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
        )
    ))
    dbutils.notebook.exit(f"Success: {result}")
except Exception as e:
    dbutils.notebook.exit(f"Error: {type(e).__name__}: {str(e)[:500]}")
