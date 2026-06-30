# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

catalog = "workspace"
schema = "mlpabf12520"
table_name = "incremental3526e9"
full_table_name = f"{catalog}.{schema}.{table_name}"
online_store_name = "mlpabf12520-online-store"

# COMMAND ----------

count = spark.sql(f"SELECT COUNT(*) as cnt FROM {full_table_name}").first()["cnt"]
print(f"Feature table has {count} rows")

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

ft = fe.get_table(name=full_table_name)
print(f"Feature table id: {ft.table_id}, name: {ft.name}")

# COMMAND ----------

# List available attributes on fe
attrs = [a for a in dir(fe) if not a.startswith('_')]
print("FeatureEngineeringClient methods:", attrs)

# COMMAND ----------

# Attempt publish_table
try:
    result = fe.publish_table(
        name=full_table_name,
        online_store=online_store_name
    )
    print(f"Published: {result}")
except AttributeError as e:
    print(f"AttributeError: {e}")
except Exception as e:
    print(f"Error type: {type(e).__name__}, msg: {e}")
