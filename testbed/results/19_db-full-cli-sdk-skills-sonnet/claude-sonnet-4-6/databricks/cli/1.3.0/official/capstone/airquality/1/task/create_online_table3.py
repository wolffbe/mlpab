# Databricks notebook source

# COMMAND ----------
import json, inspect
results = {}

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import catalog as cat

    w = WorkspaceClient()

    # Inspect the create method signature
    sig = inspect.signature(w.online_tables.create)
    results["create_sig"] = str(sig)

    # Check what OnlineTable dataclass looks like
    import databricks.sdk.service.catalog as c
    results["OnlineTable_fields"] = [f for f in dir(c.OnlineTable) if not f.startswith("_")]

except Exception as e:
    results["status"] = "error"
    results["error"] = str(e)

dbutils.notebook.exit(json.dumps(results))
