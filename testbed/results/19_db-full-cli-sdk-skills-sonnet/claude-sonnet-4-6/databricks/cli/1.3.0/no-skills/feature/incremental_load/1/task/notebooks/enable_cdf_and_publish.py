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

# Enable Change Data Feed on the feature table (required for TRIGGERED publish mode)
try:
    spark.sql(f"""
        ALTER TABLE {full_name}
        SET TBLPROPERTIES ('delta.enableChangeDataFeed' = true)
    """)
    results['cdf_enabled'] = True
    print("Change Data Feed enabled")
except Exception as e:
    results['cdf_error'] = str(e)
    print(f"CDF error: {e}")

# COMMAND ----------

# Get online store
try:
    online_store = fe.get_online_store(name=online_store_name)
    print(f"Online store state: {online_store.state}")
except Exception as e:
    results['get_online_store_error'] = str(e)
    print(f"Error: {e}")

# COMMAND ----------

# Publish to online store
try:
    publish_result = fe.publish_table(
        online_store=online_store,
        source_table_name=full_name,
        online_table_name=online_table_name,
        publish_mode="TRIGGERED"
    )
    results['publish_result'] = str(publish_result)
    print(f"Published: {publish_result}")
except Exception as e:
    results['publish_error_triggered'] = str(e)
    print(f"TRIGGERED publish error: {e}")

    # Try SNAPSHOT mode as fallback
    try:
        publish_result = fe.publish_table(
            online_store=online_store,
            source_table_name=full_name,
            online_table_name=online_table_name,
            publish_mode="SNAPSHOT"
        )
        results['publish_result_snapshot'] = str(publish_result)
        print(f"SNAPSHOT published: {publish_result}")
    except Exception as e2:
        results['publish_error_snapshot'] = str(e2)
        print(f"SNAPSHOT publish error: {e2}")

dbutils.notebook.exit(json.dumps(results, indent=2)[:3000])
