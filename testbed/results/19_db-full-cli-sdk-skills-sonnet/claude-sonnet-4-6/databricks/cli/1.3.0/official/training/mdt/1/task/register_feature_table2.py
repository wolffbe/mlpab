# Databricks notebook source
# COMMAND ----------
# Check available feature store modules
import subprocess
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'feature' in line.lower() or 'databricks' in line.lower():
        print(line)

# COMMAND ----------
# Try feature_store package
try:
    from databricks.feature_store import FeatureStoreClient
    print("feature_store available")
    fs = FeatureStoreClient()
except Exception as e:
    print(f"feature_store error: {e}")

try:
    from databricks import feature_engineering
    print("feature_engineering available")
except Exception as e:
    print(f"feature_engineering error: {e}")

# COMMAND ----------
# Try adding primary key constraint via SQL
try:
    spark.sql("""
        ALTER TABLE workspace.mlpabc8d80a.scaled7ecfaf
        ADD CONSTRAINT pk_row_id PRIMARY KEY (row_id)
    """)
    print("Primary key constraint added")
except Exception as e:
    print(f"Primary key error: {e}")

# COMMAND ----------
# Check table properties
result = spark.sql("DESCRIBE EXTENDED workspace.mlpabc8d80a.scaled7ecfaf")
result.show(50, truncate=False)
