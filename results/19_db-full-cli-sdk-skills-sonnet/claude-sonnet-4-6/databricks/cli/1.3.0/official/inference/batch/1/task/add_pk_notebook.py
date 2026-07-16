# Databricks notebook source
# COMMAND ----------
# Add primary key constraint to mark account_id as the record key
# and also try to set table properties

from pyspark.sql import functions as F

# Set table properties to indicate this is a feature table with account_id as primary key
spark.sql("""
    ALTER TABLE workspace.mlpab6ef9cb.scores4f5893
    SET TBLPROPERTIES (
        'databricks.feature_store.record_key' = 'account_id',
        'databricks.feature_store.enabled' = 'true'
    )
""")

print("Table properties set")

# Verify
df = spark.sql("SHOW TBLPROPERTIES workspace.mlpab6ef9cb.scores4f5893")
df.show(truncate=False)

dbutils.notebook.exit("Properties set successfully")
