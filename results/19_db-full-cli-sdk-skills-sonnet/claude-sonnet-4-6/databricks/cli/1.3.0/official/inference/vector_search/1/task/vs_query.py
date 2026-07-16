# Databricks notebook source
# COMMAND ----------

import json
import csv
import os
import requests

VOLUME_PATH = "/Volumes/workspace/mlpab3fd1e9/mlpab3fd1e9data"
INDEX_NAME = "workspace.mlpab3fd1e9.items9110bd"
RESULTS_PATH = f"{VOLUME_PATH}/answers.json"

# Read queries from volume
queries = []
with open(f"{VOLUME_PATH}/queries.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        queries.append({
            "query_id": row["query_id"],
            "embedding": json.loads(row["embedding"])
        })
print(f"Loaded {len(queries)} queries")

# COMMAND ----------

# Get workspace host and token from Databricks context
# In a notebook, these are available via environment
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", spark.conf.get("spark.databricks.workspaceUrl"))
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

if not DATABRICKS_HOST.startswith("https://"):
    DATABRICKS_HOST = f"https://{DATABRICKS_HOST}"

print(f"Host: {DATABRICKS_HOST}")
print(f"Index: {INDEX_NAME}")

# COMMAND ----------

# Query all queries via REST API
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

encoded_index = INDEX_NAME.replace(".", "%2E").replace("/", "%2F")
# Use the URL-encoded index name directly as path segment
url = f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{INDEX_NAME}/query"

neighbors = {}
for q in queries:
    payload = {
        "query_vector": q["embedding"],
        "columns": ["item_id"],
        "num_results": 5
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "result" in data and "data_array" in data["result"] and data["result"]["data_array"]:
            item_ids = [row[0] for row in data["result"]["data_array"]]
        else:
            item_ids = []
            print(f"  WARNING: No results for {q['query_id']}, response: {data}")

        neighbors[q["query_id"]] = item_ids
        print(f"  Query {q['query_id']}: {item_ids}")
    except Exception as e:
        print(f"  ERROR for query {q['query_id']}: {e}")
        neighbors[q["query_id"]] = []

# COMMAND ----------

# Write results to volume
answers = {
    "store": "workspace.mlpab3fd1e9.items9110bd",
    "neighbors": neighbors
}

with open(RESULTS_PATH, "w") as f:
    json.dump(answers, f, indent=2)

print(f"\nResults written to {RESULTS_PATH}")
print(json.dumps(answers, indent=2))
