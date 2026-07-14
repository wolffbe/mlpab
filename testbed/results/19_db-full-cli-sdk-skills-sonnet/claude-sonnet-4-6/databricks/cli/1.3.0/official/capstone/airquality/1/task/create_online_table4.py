# Databricks notebook source

# COMMAND ----------
import json
results = {}

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import (
        OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
    )

    w = WorkspaceClient()
    full_schema = "workspace.mlpab375647"
    pred_name = "airqpredfdfb59"
    online_table_name = f"{full_schema}.{pred_name}_online"

    table = OnlineTable(
        name=online_table_name,
        spec=OnlineTableSpec(
            source_table_full_name=f"{full_schema}.{pred_name}",
            primary_key_columns=["date"],
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
        )
    )

    result = w.online_tables.create(table)
    results["status"] = "created"
    results["name"] = online_table_name
    results["result"] = str(result)[:500]
except Exception as e:
    results["status"] = "error"
    results["error"] = str(e)[:500]
    results["error_type"] = type(e).__name__

dbutils.notebook.exit(json.dumps(results))
