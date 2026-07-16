# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

attrs = [a for a in dir(fe) if not a.startswith('_')]
output = "Methods: " + str(attrs)

# Try publish_table
full_table_name = "workspace.mlpabf12520.incremental3526e9"
online_store_name = "mlpabf12520-online-store"

result_str = ""
try:
    result = fe.publish_table(
        name=full_table_name,
        online_store=online_store_name
    )
    result_str = f"SUCCESS: {result}"
except AttributeError as e:
    result_str = f"AttributeError (no publish_table): {e}"
except Exception as e:
    result_str = f"Error: {type(e).__name__}: {e}"

import json
dbutils.notebook.exit(json.dumps({"methods": attrs, "publish_result": result_str}))
