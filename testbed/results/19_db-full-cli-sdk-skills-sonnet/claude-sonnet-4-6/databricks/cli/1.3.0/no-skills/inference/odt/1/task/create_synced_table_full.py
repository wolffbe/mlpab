# Databricks notebook source
# COMMAND ----------
import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Step 1: Try to register the Postgres database as a UC catalog
# Using various possible field names
catalog_id = "mlpab08bf79uccat"
for field_name, field_value in [
    ("database", "projects/mlpab08bf79-ccpred/branches/production/databases/databricks-postgres"),
    ("parent_database", "projects/mlpab08bf79-ccpred/branches/production/databases/databricks-postgres"),
    ("lakebase_database", "projects/mlpab08bf79-ccpred/branches/production/databases/databricks-postgres"),
    ("source", "projects/mlpab08bf79-ccpred/branches/production/databases/databricks-postgres"),
]:
    body = {"catalog": {field_name: field_value}}
    resp = requests.post(
        f"https://{host}/api/2.0/postgres/catalogs?catalog_id={catalog_id}",
        json=body,
        headers=headers
    )
    results[f"catalog_{field_name}"] = f"{resp.status_code}: {resp.text[:200]}"
    if resp.status_code == 200:
        results["catalog_success_field"] = field_name
        break

# Also try synced table with catalog reference
for field_name, field_value in [
    ("catalog", f"catalogs/{catalog_id}"),
    ("lakebase_catalog", f"catalogs/{catalog_id}"),
    ("target_catalog", f"catalogs/{catalog_id}"),
]:
    body = {"synced_table": {field_name: field_value}}
    resp = requests.post(
        f"https://{host}/api/2.0/postgres/synced_tables?synced_table_id=workspace.mlpaba35f2a.scored50223c",
        json=body,
        headers=headers
    )
    results[f"synced_{field_name}"] = f"{resp.status_code}: {resp.text[:200]}"
    if resp.status_code not in [400] or "non-default" not in resp.text:
        results[f"synced_{field_name}_maybe"] = True
        break

dbutils.notebook.exit(json.dumps(results))
