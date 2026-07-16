# Databricks notebook source
# COMMAND ----------
import json, requests, subprocess

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# Try to get the pipeline for the existing synced table
try:
    r = requests.get(
        f"{host}/api/2.0/pipelines/f3ff9421-cc65-4e9a-a79f-201921b0b67b",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['pipeline'] = {"status": r.status_code, "body": r.text[:1000]}
except Exception as e:
    results['pipeline'] = {"error": str(e)}

# Let's try to understand what creates a synced table
# The synced table endpoint has two parts:
# 1. Creates a DLT pipeline to sync data
# 2. Creates a FOREIGN table in UC (postgres format)

# Try the create with different content-type or headers
payloads_to_try = [
    # Minimal payload - just the source table name inside the synced_table object
    json.dumps({"synced_table": {"source_table_name": "workspace.mlpabf1452c.featuresb1ea93"}}),
    # Try with version
    json.dumps({"synced_table": {"version": 1}}),
    # Try nested spec
    json.dumps({"synced_table": {"spec": {"primary_key_columns": ["row_id"]}}}),
]

for i, payload_str in enumerate(payloads_to_try):
    try:
        r = requests.post(
            f"{host}/api/2.0/postgres/synced_tables?synced_table_id=workspace.mlpabf1452c.featuresb1ea93_online",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            data=payload_str,
            timeout=30
        )
        results[f'post_{i}'] = {"status": r.status_code, "body": r.text[:500]}
        if r.status_code in (200, 201):
            break
    except Exception as e:
        results[f'post_{i}'] = {"error": str(e)}

# Also try looking at what the scaledd437a3 source table is
try:
    r_src = requests.get(
        f"{host}/api/2.1/unity-catalog/tables/workspace.mlpabc1d5e2.scaledd437a3",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['scaledd437a3_src'] = {"status": r_src.status_code, "body": r_src.text[:800]}
except Exception as e:
    results['scaledd437a3_src'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
