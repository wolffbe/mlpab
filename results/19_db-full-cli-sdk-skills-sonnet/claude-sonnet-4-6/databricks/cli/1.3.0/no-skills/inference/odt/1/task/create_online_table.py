# Databricks notebook source
# COMMAND ----------
import json
import inspect
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec

results = {}

# Inspect OnlineTableSpec
results["spec_fields"] = str(inspect.signature(OnlineTableSpec.__init__))
results["spec_source"] = str([f for f in OnlineTableSpec.__dataclass_fields__.keys()] if hasattr(OnlineTableSpec, '__dataclass_fields__') else "not dataclass")

w = WorkspaceClient()

# Try with minimal spec - just source and primary key
try:
    spec = OnlineTableSpec(
        source_table_full_name="workspace.mlpaba35f2a.scored50223c",
        primary_key_columns=["request_id"]
    )
    results["spec_repr"] = str(spec)

    result = w.online_tables.create(
        name="workspace.mlpaba35f2a.scored50223c_online",
        spec=spec
    )
    results["create_result"] = str(result)[:500]
except Exception as e:
    results["create_error"] = str(e)[:500]

    # Try with dict spec directly
    try:
        import requests
        host = spark.conf.get("spark.databricks.workspaceUrl")
        token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        payload = {
            "name": "workspace.mlpaba35f2a.scored50223c_online",
            "spec": {
                "source_table_full_name": "workspace.mlpaba35f2a.scored50223c",
                "primary_key_columns": ["request_id"],
                "run_triggered": {"triggered_updates": {}}
            }
        }
        resp = requests.post(f"https://{host}/api/2.0/online-tables", json=payload, headers=headers)
        results["raw_post"] = f"{resp.status_code}: {resp.text[:300]}"
    except Exception as e2:
        results["raw_post_error"] = str(e2)[:300]

dbutils.notebook.exit(json.dumps(results))
