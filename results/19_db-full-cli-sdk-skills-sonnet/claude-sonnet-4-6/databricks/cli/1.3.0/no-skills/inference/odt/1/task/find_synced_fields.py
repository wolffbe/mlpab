# Databricks notebook source
# COMMAND ----------
import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Try the raw API with different JSON structures for synced_table
for body_variant, body in [
    ("basic_delta", {"synced_table": {"delta_table": {"source_table_full_name": "workspace.mlpaba35f2a.scored50223c", "primary_key_columns": ["request_id"]}}}),
    ("basic_source", {"synced_table": {"source_table_full_name": "workspace.mlpaba35f2a.scored50223c", "primary_key_columns": ["request_id"]}}),
    ("delta_source", {"synced_table": {"delta_source": {"full_name": "workspace.mlpaba35f2a.scored50223c"}}}),
    ("unity_catalog", {"synced_table": {"unity_catalog_table": "workspace.mlpaba35f2a.scored50223c", "primary_key_columns": ["request_id"]}}),
    ("snapshot_spec", {"synced_table": {"snapshot_spec": {"source_table_full_name": "workspace.mlpaba35f2a.scored50223c", "primary_key_columns": ["request_id"]}}}),
    ("table_spec", {"synced_table": {"table_spec": {"source_table_full_name": "workspace.mlpaba35f2a.scored50223c"}}}),
    ("minimal", {"synced_table": {"name": "workspace.mlpaba35f2a.scored50223c"}}),
]:
    resp = requests.post(
        f"https://{host}/api/2.0/postgres/synced_tables?synced_table_id=workspace.mlpaba35f2a.scored50223c",
        json=body,
        headers=headers
    )
    results[body_variant] = f"{resp.status_code}: {resp.text[:300]}"

dbutils.notebook.exit(json.dumps(results))
