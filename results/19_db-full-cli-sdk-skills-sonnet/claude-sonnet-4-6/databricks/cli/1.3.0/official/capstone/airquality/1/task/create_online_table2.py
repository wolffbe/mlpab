# Databricks notebook source

# COMMAND ----------
import json
results = {}

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

    w = WorkspaceClient()
    full_schema = "workspace.mlpab375647"
    pred_name = "airqpredfdfb59"
    online_table_name = f"{full_schema}.{pred_name}_online"

    spec = OnlineTableSpec(
        source_table_full_name=f"{full_schema}.{pred_name}",
        primary_key_columns=["date"],
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
    )

    result = w.online_tables.create(name=online_table_name, spec=spec)
    results["status"] = "created"
    results["name"] = online_table_name
    results["result"] = str(result)
except Exception as e:
    results["status"] = "error"
    results["error"] = str(e)
    results["error_type"] = type(e).__name__

dbutils.notebook.exit(json.dumps(results))
