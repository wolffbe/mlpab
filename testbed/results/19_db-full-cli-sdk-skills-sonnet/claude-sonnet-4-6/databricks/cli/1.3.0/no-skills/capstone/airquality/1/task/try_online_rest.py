# Databricks notebook source

# COMMAND ----------

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()

import requests, json
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

with open("/Volumes/workspace/mlpabd7768b/airqdata/online_rest_result.txt", "w") as f:
    # Try creating online table directly via REST (bypassing SDK/CLI deprecation check)
    body = {
        "name": "workspace.mlpabd7768b.airqpredfdfb59_online",
        "spec": {
            "source_table_full_name": "workspace.mlpabd7768b.airqpredfdfb59",
            "primary_key_columns": ["date"],
            "run_triggered": {}
        }
    }
    resp = requests.post(f"{host}/api/2.0/online-tables", headers=headers, json=body)
    f.write(f"POST /online-tables: {resp.status_code}\n{resp.text[:500]}\n\n")

    # Try v1
    resp = requests.post(f"{host}/api/1.0/online-tables", headers=headers, json=body)
    f.write(f"POST /api/1.0/online-tables: {resp.status_code}\n{resp.text[:300]}\n\n")

    # Check if there's any way to query or list synced tables via GET
    resp = requests.get(f"{host}/api/2.0/online-tables", headers=headers)
    f.write(f"GET /online-tables: {resp.status_code}\n{resp.text[:300]}\n\n")

    # Try getting the online table catalog entry
    resp = requests.get(f"{host}/api/2.1/unity-catalog/tables/workspace.mlpabd7768b.airqpredfdfb59", headers=headers)
    f.write(f"GET UC table: {resp.status_code}\n")
    if resp.ok:
        tbl = resp.json()
        f.write(f"table_type: {tbl.get('table_type')}\n")
        f.write(f"data_source_format: {tbl.get('data_source_format')}\n")

print("Done")
dbutils.notebook.exit("done")
