# Databricks notebook source

# COMMAND ----------
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

try:
    result = w.online_tables.create(name=online_table_name, spec=spec)
    print(f"Online table created: {online_table_name}")
    print(result)
except Exception as e:
    print(f"Online table creation error: {e}")
    # Try synced table approach
    try:
        result2 = w.api_client.do(
            "POST",
            "/api/2.0/online-tables",
            body={
                "name": online_table_name,
                "spec": {
                    "source_table_full_name": f"{full_schema}.{pred_name}",
                    "primary_key_columns": ["date"],
                    "run_triggered": {}
                }
            }
        )
        print(f"Result2: {result2}")
    except Exception as e2:
        print(f"Second attempt failed: {e2}")
