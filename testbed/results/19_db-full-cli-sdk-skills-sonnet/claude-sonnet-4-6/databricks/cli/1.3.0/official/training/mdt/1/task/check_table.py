# Databricks notebook source
# COMMAND ----------
# Check what packages are available for feature store
import pkg_resources
packages = [d.project_name for d in pkg_resources.working_set]
fs_packages = [p for p in packages if 'feature' in p.lower() or 'databricks' in p.lower()]
for p in sorted(fs_packages):
    print(p)

# COMMAND ----------
# Check if primary key was set on the table
result = spark.sql("DESCRIBE EXTENDED workspace.mlpabc8d80a.scaled7ecfaf")
for row in result.collect():
    print(row)

# COMMAND ----------
# Check table constraints
try:
    result2 = spark.sql("SHOW CONSTRAINTS ON workspace.mlpabc8d80a.scaled7ecfaf")
    result2.show(truncate=False)
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------
# Try feature_store client
try:
    from databricks.feature_store import FeatureStoreClient
    print("feature_store available")
    fs = FeatureStoreClient()
    ft = fs.get_table("workspace.mlpabc8d80a.scaled7ecfaf")
    print(f"Table: {ft}")
except Exception as e:
    print(f"feature_store error: {e}")

dbutils.notebook.exit("done")
