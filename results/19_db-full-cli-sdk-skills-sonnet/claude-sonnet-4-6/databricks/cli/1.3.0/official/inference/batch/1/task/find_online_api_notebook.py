# Databricks notebook source
# COMMAND ----------
import requests, json

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# Try more paths
more_paths = [
    ("POST", "/api/2.0/online-tables", {
        "name": "workspace.mlpab6ef9cb.scores4f5893_online",
        "spec": {
            "source_table_full_name": "workspace.mlpab6ef9cb.scores4f5893",
            "primary_key_columns": ["account_id"],
            "run_triggered": {},
            "perform_full_copy": True
        }
    }),
    # Try to get current list of online tables
    ("GET", "/api/2.0/online-tables/workspace.mlpab6ef9cb.scores4f5893", {}),
    # Try feature store API
    ("GET", "/api/2.0/feature-store/feature-tables", {}),
    ("POST", "/api/2.0/feature-store/feature-tables", {
        "name": "workspace.mlpab6ef9cb.scores4f5893",
        "primary_keys": [{"name": "account_id", "data_type": "STRING"}]
    }),
]

for method, path, payload in more_paths:
    url = f"https://{host}{path}"
    if method == "POST":
        r = requests.post(url, json=payload, headers=headers)
    else:
        r = requests.get(url, headers=headers)
    results.append(f"{method} {path}: {r.status_code} {r.text[:200] if r.status_code != 404 else ''}")

# Try via catalog API with special table_type
# Also check if there's a way to get a serving/online table through SQL
sql_check = """
SELECT * FROM workspace.mlpab6ef9cb.scores4f5893 LIMIT 3
"""
df = spark.sql(sql_check)
results.append(f"\nScores table has {df.count()} rows")
results.append(f"Sample: {df.collect()[:2]}")

dbutils.notebook.exit("\n".join(results))
