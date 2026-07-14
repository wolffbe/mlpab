# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.mlpab6c9d6b.training AS
# MAGIC SELECT * FROM csv."`dbfs:/Volumes/workspace/mlpab6c9d6b/skew_volume/training_sample.csv`"
# MAGIC WITH (header = true, inferSchema = true);

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.mlpab6c9d6b.serving AS
# MAGIC SELECT * FROM csv."`dbfs:/Volumes/workspace/mlpab6c9d6b/skew_volume/serving_log.csv`"
# MAGIC WITH (header = true, inferSchema = true);

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Compute summary statistics for each feature in training
# MAGIC CREATE OR REPLACE TABLE workspace.mlpab6c9d6b.training_stats AS
# MAGIC SELECT
# MAGIC   feature,
# MAGIC   AVG(value) AS mean,
# MAGIC   STDDEV(value) AS stddev,
# MAGIC   MIN(value) AS min,
# MAGIC   MAX(value) AS max
# MAGIC FROM (
# MAGIC   SELECT 'f1' AS feature, f1 AS value FROM workspace.mlpab6c9d6b.training
# MAGIC   UNION ALL SELECT 'f2' AS feature, f2 AS value FROM workspace.mlpab6c9d6b.training
# MAGIC   UNION ALL SELECT 'f3' AS feature, f3 AS value FROM workspace.mlpab6c9d6b.training
# MAGIC   UNION ALL SELECT 'f4' AS feature, f4 AS value FROM workspace.mlpab6c9d6b.training
# MAGIC   UNION ALL SELECT 'f5' AS feature, f5 AS value FROM workspace.mlpab6c9d6b.training
# MAGIC )
# MAGIC GROUP BY feature;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Compute summary statistics for each feature in serving
# MAGIC CREATE OR REPLACE TABLE workspace.mlpab6c9d6b.serving_stats AS
# MAGIC SELECT
# MAGIC   feature,
# MAGIC   AVG(value) AS mean,
# MAGIC   STDDEV(value) AS stddev,
# MAGIC   MIN(value) AS min,
# MAGIC   MAX(value) AS max
# MAGIC FROM (
# MAGIC   SELECT 'f1' AS feature, f1 AS value FROM workspace.mlpab6c9d6b.serving
# MAGIC   UNION ALL SELECT 'f2' AS feature, f2 AS value FROM workspace.mlpab6c9d6b.serving
# MAGIC   UNION ALL SELECT 'f3' AS feature, f3 AS value FROM workspace.mlpab6c9d6b.serving
# MAGIC   UNION ALL SELECT 'f4' AS feature, f4 AS value FROM workspace.mlpab6c9d6b.serving
# MAGIC   UNION ALL SELECT 'f5' AS feature, f5 AS value FROM workspace.mlpab6c9d6b.serving
# MAGIC )
# MAGIC GROUP BY feature;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Join and compare statistics
# MAGIC CREATE OR REPLACE TABLE workspace.mlpab6c9d6b.feature_comparison AS
# MAGIC SELECT
# MAGIC   t.feature,
# MAGIC   t.mean AS training_mean,
# MAGIC   s.mean AS serving_mean,
# MAGIC   ABS(t.mean - s.mean) AS mean_diff,
# MAGIC   t.stddev AS training_stddev,
# MAGIC   s.stddev AS serving_stddev,
# MAGIC   ABS(t.stddev - s.stddev) AS stddev_diff,
# MAGIC   t.min AS training_min,
# MAGIC   s.min AS serving_min,
# MAGIC   t.max AS training_max,
# MAGIC   s.max AS serving_max
# MAGIC FROM workspace.mlpab6c9d6b.training_stats t
# MAGIC JOIN workspace.mlpab6c9d6b.serving_stats s
# MAGIC ON t.feature = s.feature;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM workspace.mlpab6c9d6b.feature_comparison ORDER BY mean_diff DESC;