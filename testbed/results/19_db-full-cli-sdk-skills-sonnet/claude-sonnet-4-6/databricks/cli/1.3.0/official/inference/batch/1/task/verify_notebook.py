# Databricks notebook source
# COMMAND ----------
df = spark.sql("SELECT COUNT(*) as cnt FROM workspace.mlpab6ef9cb.scores4f5893")
count = df.collect()[0]['cnt']

df2 = spark.sql("SELECT * FROM workspace.mlpab6ef9cb.scores4f5893 ORDER BY account_id LIMIT 5")
samples = df2.collect()

result = f"Row count: {count}\nSamples: {samples}"
print(result)
dbutils.notebook.exit(result)
