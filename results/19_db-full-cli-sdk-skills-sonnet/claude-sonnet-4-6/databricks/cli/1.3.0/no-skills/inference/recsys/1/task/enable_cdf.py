# Databricks notebook source
# COMMAND ----------
# Enable Change Data Feed on the feature table
table_name = "workspace.mlpabb40f43.recs708df6"

spark.sql(f"""
ALTER TABLE {table_name}
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

print(f"Change Data Feed enabled on {table_name}")

# Verify
result = spark.sql(f"DESCRIBE DETAIL {table_name}").collect()[0]
print(f"Table properties: {result}")

spark.sql(f"SHOW TBLPROPERTIES {table_name}").show(20, truncate=False)

dbutils.notebook.exit("CDF enabled successfully")
