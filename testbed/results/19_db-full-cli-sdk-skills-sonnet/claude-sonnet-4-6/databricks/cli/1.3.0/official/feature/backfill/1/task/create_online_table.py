# Create online table for feature table accountse81ff1
import requests
import os

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Use the feature-store publish-table REST API
url = f"https://{host}/api/2.0/feature-store/tables/workspace.mlpab0442b8.accountse81ff1/publish"
payload = {
    "publish_spec": {
        "online_store": "mlpab0442b8-online-store",
        "online_table_name": "accountse81ff1",
        "publish_mode": 1
    }
}
response = requests.post(url, headers=headers, json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
