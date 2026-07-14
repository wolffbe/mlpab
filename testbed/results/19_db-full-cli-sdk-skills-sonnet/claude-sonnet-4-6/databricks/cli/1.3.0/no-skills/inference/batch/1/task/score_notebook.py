# Databricks notebook source

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.mlpab685a31.scores4f5893 AS
# MAGIC WITH raw AS (
# MAGIC   SELECT
# MAGIC     account_id,
# MAGIC     CAST(event_time AS BIGINT) as event_time,
# MAGIC     CAST(f1 AS DOUBLE) as f1,
# MAGIC     CAST(f2 AS DOUBLE) as f2,
# MAGIC     CAST(f3 AS DOUBLE) as f3
# MAGIC   FROM read_files('/Volumes/workspace/mlpab685a31/data_files/feature_history.csv',
# MAGIC     format => 'csv', header => true)
# MAGIC   WHERE CAST(event_time AS BIGINT) <= 1773306000000
# MAGIC ),
# MAGIC ranked AS (
# MAGIC   SELECT *,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
# MAGIC   FROM raw
# MAGIC )
# MAGIC SELECT
# MAGIC   account_id,
# MAGIC   ROUND(1.0 / (1.0 + EXP(-(-0.9682 * f1 + (-0.0299) * f2 + 1.2708 * f3 + (-0.1715)))), 6) as score
# MAGIC FROM ranked
# MAGIC WHERE rn = 1
