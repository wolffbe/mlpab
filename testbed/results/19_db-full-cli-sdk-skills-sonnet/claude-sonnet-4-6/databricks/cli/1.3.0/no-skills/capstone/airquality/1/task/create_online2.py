# Databricks notebook source

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as cat_svc
import inspect

w = WorkspaceClient()

# Inspect the create method signature
sig = inspect.signature(w.online_tables.create)
results = [f"create signature: {sig}"]

with open("/Volumes/workspace/mlpabd7768b/airqdata/online_table_result.txt", "w") as f:
    f.write(f"create signature: {sig}\n")

    # Try creating an OnlineTable object first
    try:
        online_table = cat_svc.OnlineTable(
            name="workspace.mlpabd7768b.airqpredfdfb59_online",
            spec=cat_svc.OnlineTableSpec(
                source_table_full_name="workspace.mlpabd7768b.airqpredfdfb59",
                primary_key_columns=["date"],
                run_triggered=cat_svc.OnlineTableSpecTriggeredSchedulingPolicy()
            )
        )
        f.write(f"OnlineTable object: {online_table}\n")

        result = w.online_tables.create(online_table)
        f.write(f"Created: {result}\n")
    except Exception as e:
        f.write(f"Attempt 1 failed: {e}\n")

    # Try with table keyword
    try:
        result = w.online_tables.create(table=cat_svc.OnlineTable(
            name="workspace.mlpabd7768b.airqpredfdfb59_online",
            spec=cat_svc.OnlineTableSpec(
                source_table_full_name="workspace.mlpabd7768b.airqpredfdfb59",
                primary_key_columns=["date"],
                run_triggered=cat_svc.OnlineTableSpecTriggeredSchedulingPolicy()
            )
        ))
        f.write(f"Created with table kwarg: {result}\n")
    except Exception as e:
        f.write(f"Attempt 2 failed: {e}\n")

print("Done - check /Volumes/workspace/mlpabd7768b/airqdata/online_table_result.txt")
dbutils.notebook.exit("done")
