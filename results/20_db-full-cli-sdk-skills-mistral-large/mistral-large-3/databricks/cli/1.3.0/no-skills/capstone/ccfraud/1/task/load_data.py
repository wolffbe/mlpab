# Databricks notebook source
# MAGIC %md
# MAGIC ## Load Transaction Data into Unity Catalog

# COMMAND ----------

# Create transactions table
spark.read.csv("/dbfs${PWD}/data/transactions.csv", header=True, inferSchema=True).write.saveAsTable("workspace.${MLPAB_DATABRICKS_SCHEMA}.transactions")

# Create score_transactions table
spark.read.csv("/dbfs${PWD}/data/score_transactions.csv", header=True, inferSchema=True).write.saveAsTable("workspace.${MLPAB_DATABRICKS_SCHEMA}.score_transactions")

# COMMAND ----------

# Verify tables
display(spark.sql("SHOW TABLES IN workspace.${MLPAB_DATABRICKS_SCHEMA}"))