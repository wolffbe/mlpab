# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering.client import DatabricksOnlineStore

fe = FeatureEngineeringClient()

full_table_name = "workspace.mlpabf12520.incremental3526e9"
online_store_name = "mlpabf12520-online-store"

# DatabricksOnlineStore wraps the Databricks online feature store
online_store = DatabricksOnlineStore(name=online_store_name, capacity="CU_1")

result = fe.publish_table(
    name=full_table_name,
    online_store=online_store,
    online_table_name="workspace.mlpabf12520.incremental3526e9_pub"
)
print(f"Publish result: {result}")

import json
dbutils.notebook.exit(json.dumps({"status": "published", "online_store": online_store_name, "result": str(result)}))
