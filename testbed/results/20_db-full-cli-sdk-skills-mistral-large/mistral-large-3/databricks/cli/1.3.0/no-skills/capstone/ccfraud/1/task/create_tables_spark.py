# Databricks notebook source
# MAGIC %md
# MAGIC ## Create Tables in Unity Catalog

# COMMAND ----------

# Create transactions table
df_transactions = spark.read.csv("/dbfs${PWD}/data/transactions.csv", header=True, inferSchema=True)
df_transactions.write.saveAsTable("workspace.${MLPAB_DATABRICKS_SCHEMA}.transactions")

# Create score_transactions table
df_score = spark.read.csv("/dbfs${PWD}/data/score_transactions.csv", header=True, inferSchema=True)
df_score.write.saveAsTable("workspace.${MLPAB_DATABRICKS_SCHEMA}.score_transactions")

# Verify tables
display(spark.sql("SHOW TABLES IN workspace.${MLPAB_DATABRICKS_SCHEMA}"))