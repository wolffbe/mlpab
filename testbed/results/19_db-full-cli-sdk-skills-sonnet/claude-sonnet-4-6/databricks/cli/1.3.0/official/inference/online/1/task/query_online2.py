# Databricks notebook source
# COMMAND ----------
import json, time
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
base_url = f"https://{host}"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

schema = "workspace.mlpabae7d2f"
table_name = f"{schema}.profilesaa70e4"
online_store_name = "mlpabae7d2f-store"
online_table_name = "profilesaa70e4_online"
online_table_full = f"{schema}.{online_table_name}"

lookup_keys = [
    "A0003", "A0005", "A0012", "A0015", "A0023", "A0030", "A0031", "A0034",
    "A0048", "A0049", "A0055", "A0063", "A0066", "A0072", "A0085", "A0090",
    "A0103", "A0109", "A0112", "A0113"
]

diag = {}

# Check online store state
r = requests.get(f"{base_url}/api/2.0/feature-store/online-stores/{online_store_name}", headers=headers)
diag["online_store"] = {"status": r.status_code, "body": r.text[:500]}

# List feature tables in online store
r = requests.get(f"{base_url}/api/2.0/feature-store/online-stores/{online_store_name}/feature-tables", headers=headers)
diag["feature_tables"] = {"status": r.status_code, "body": r.text[:1000]}

# List all serving endpoints
r = requests.get(f"{base_url}/api/2.0/serving-endpoints", headers=headers)
diag["serving_endpoints"] = {"status": r.status_code, "body": r.text[:2000]}

# Try UC online tables
r = requests.get(f"{base_url}/api/2.1/unity-catalog/online-tables", headers=headers)
diag["uc_online_tables"] = {"status": r.status_code, "body": r.text[:2000]}

# Try lookup endpoint
test_key = "A0003"
for path in [
    f"/api/2.0/feature-store/online-stores/{online_store_name}/feature-tables/{table_name}/lookup",
    f"/api/2.0/feature-store/online/{table_name}",
    f"/api/2.0/feature-store/tables/{table_name}/online-lookup",
    f"/api/2.0/feature-store/online-stores/{online_store_name}/online-tables/{online_table_full}/lookup",
]:
    try:
        r = requests.post(f"{base_url}{path}", headers=headers, json={"account_id": test_key}, timeout=10)
        diag[f"lookup_{path[-30:]}"] = {"status": r.status_code, "body": r.text[:400]}
    except Exception as e:
        diag[f"lookup_{path[-30:]}"] = {"error": str(e)}

# Also try reading the online table via SQL
try:
    rows = spark.sql(f"SELECT * FROM {online_table_full} WHERE account_id IN ('{test_key}')").collect()
    diag["online_table_sql"] = [{"account_id": r.account_id, "f1": r.f1} for r in rows]
except Exception as e:
    diag["online_table_sql"] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(diag))
