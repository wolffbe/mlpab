# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

catalog_name = "workspace"
schema_name = "mlpab312fe6"
table_name = "incremental3526e9"
full_name = f"{catalog_name}.{schema_name}.{table_name}"
online_store_name = "mlpab312fe6-online"
online_table_name = f"{catalog_name}.{schema_name}.incremental3526e9_online"

results = {}

# COMMAND ----------

# Get the existing online store
try:
    online_store = fe.get_online_store(name=online_store_name)
    results['online_store_state'] = str(online_store)
    print(f"Online store: {online_store}")
except Exception as e:
    results['get_online_store_error'] = str(e)
    print(f"Get online store error: {e}")

# COMMAND ----------

# Publish feature table to online store
try:
    publish_result = fe.publish_table(
        online_store=online_store,
        source_table_name=full_name,
        online_table_name=online_table_name,
        publish_mode="TRIGGERED"
    )
    results['publish_result'] = str(publish_result)
    print(f"Table published: {publish_result}")
except Exception as e:
    results['publish_error'] = str(e)
    print(f"Publish error: {e}")

dbutils.notebook.exit(json.dumps(results, indent=2))
