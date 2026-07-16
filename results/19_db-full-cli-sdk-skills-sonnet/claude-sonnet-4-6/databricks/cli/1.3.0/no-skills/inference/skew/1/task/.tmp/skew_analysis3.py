# Databricks notebook source

# COMMAND ----------
spark.sql("DROP TABLE IF EXISTS workspace.mlpabb808b1.training")
spark.sql("DROP TABLE IF EXISTS workspace.mlpabb808b1.serving")

spark.sql("""
CREATE TABLE workspace.mlpabb808b1.training
AS SELECT * FROM read_files(
  '/Volumes/workspace/mlpabb808b1/skew_data/training_sample.csv',
  format => 'csv',
  header => true,
  inferSchema => true
)
""")

spark.sql("""
CREATE TABLE workspace.mlpabb808b1.serving
AS SELECT * FROM read_files(
  '/Volumes/workspace/mlpabb808b1/skew_data/serving_log.csv',
  format => 'csv',
  header => true,
  inferSchema => true
)
""")

print("Tables created!")

# COMMAND ----------
result = spark.sql("""
WITH joined AS (
  SELECT
    t.entity_id,
    t.f1 AS t_f1, s.f1 AS s_f1,
    t.f2 AS t_f2, s.f2 AS s_f2,
    t.f3 AS t_f3, s.f3 AS s_f3,
    t.f4 AS t_f4, s.f4 AS s_f4,
    t.f5 AS t_f5, s.f5 AS s_f5
  FROM workspace.mlpabb808b1.training t
  INNER JOIN workspace.mlpabb808b1.serving s ON t.entity_id = s.entity_id
),
diffs AS (
  SELECT
    entity_id,
    ABS(t_f1 - s_f1) AS diff_f1,
    ABS(t_f2 - s_f2) AS diff_f2,
    ABS(t_f3 - s_f3) AS diff_f3,
    ABS(t_f4 - s_f4) AS diff_f4,
    ABS(t_f5 - s_f5) AS diff_f5
  FROM joined
)
SELECT
  AVG(diff_f1) AS avg_diff_f1,
  AVG(diff_f2) AS avg_diff_f2,
  AVG(diff_f3) AS avg_diff_f3,
  AVG(diff_f4) AS avg_diff_f4,
  AVG(diff_f5) AS avg_diff_f5,
  MAX(diff_f1) AS max_diff_f1,
  MAX(diff_f2) AS max_diff_f2,
  MAX(diff_f3) AS max_diff_f3,
  MAX(diff_f4) AS max_diff_f4,
  MAX(diff_f5) AS max_diff_f5,
  COUNT(*) AS n_common_entities
FROM diffs
""")

result.show()

# COMMAND ----------
dist_result = spark.sql("""
WITH joined AS (
  SELECT
    t.entity_id,
    t.f1 AS t_f1, s.f1 AS s_f1,
    t.f2 AS t_f2, s.f2 AS s_f2,
    t.f3 AS t_f3, s.f3 AS s_f3,
    t.f4 AS t_f4, s.f4 AS s_f4,
    t.f5 AS t_f5, s.f5 AS s_f5
  FROM workspace.mlpabb808b1.training t
  INNER JOIN workspace.mlpabb808b1.serving s ON t.entity_id = s.entity_id
)
SELECT
  'training' AS source,
  AVG(t_f1) AS avg_f1, AVG(t_f2) AS avg_f2, AVG(t_f3) AS avg_f3, AVG(t_f4) AS avg_f4, AVG(t_f5) AS avg_f5,
  STDDEV(t_f1) AS std_f1, STDDEV(t_f2) AS std_f2, STDDEV(t_f3) AS std_f3, STDDEV(t_f4) AS std_f4, STDDEV(t_f5) AS std_f5
FROM joined
UNION ALL
SELECT
  'serving' AS source,
  AVG(s_f1) AS avg_f1, AVG(s_f2) AS avg_f2, AVG(s_f3) AS avg_f3, AVG(s_f4) AS avg_f4, AVG(s_f5) AS avg_f5,
  STDDEV(s_f1) AS std_f1, STDDEV(s_f2) AS std_f2, STDDEV(s_f3) AS std_f3, STDDEV(s_f4) AS std_f4, STDDEV(s_f5) AS std_f5
FROM joined
""")

dist_result.show(truncate=False)

# COMMAND ----------
ratio_stats = spark.sql("""
WITH joined AS (
  SELECT
    t.entity_id,
    t.f1 AS t_f1, s.f1 AS s_f1,
    t.f2 AS t_f2, s.f2 AS s_f2,
    t.f3 AS t_f3, s.f3 AS s_f3,
    t.f4 AS t_f4, s.f4 AS s_f4,
    t.f5 AS t_f5, s.f5 AS s_f5
  FROM workspace.mlpabb808b1.training t
  INNER JOIN workspace.mlpabb808b1.serving s ON t.entity_id = s.entity_id
)
SELECT
  AVG(CASE WHEN s_f1 != 0 THEN t_f1/s_f1 END) AS avg_ratio_f1,
  AVG(CASE WHEN s_f2 != 0 THEN t_f2/s_f2 END) AS avg_ratio_f2,
  AVG(CASE WHEN s_f3 != 0 THEN t_f3/s_f3 END) AS avg_ratio_f3,
  AVG(CASE WHEN s_f4 != 0 THEN t_f4/s_f4 END) AS avg_ratio_f4,
  AVG(CASE WHEN s_f5 != 0 THEN t_f5/s_f5 END) AS avg_ratio_f5,
  STDDEV(CASE WHEN s_f1 != 0 THEN t_f1/s_f1 END) AS std_ratio_f1,
  STDDEV(CASE WHEN s_f2 != 0 THEN t_f2/s_f2 END) AS std_ratio_f2,
  STDDEV(CASE WHEN s_f3 != 0 THEN t_f3/s_f3 END) AS std_ratio_f3,
  STDDEV(CASE WHEN s_f4 != 0 THEN t_f4/s_f4 END) AS std_ratio_f4,
  STDDEV(CASE WHEN s_f5 != 0 THEN t_f5/s_f5 END) AS std_ratio_f5
FROM joined
""")

ratio_stats.show(truncate=False)

# COMMAND ----------
spark.sql("DROP TABLE IF EXISTS workspace.mlpabb808b1.skew_results")
spark.sql("""
CREATE TABLE workspace.mlpabb808b1.skew_results AS
WITH joined AS (
  SELECT
    t.entity_id,
    t.f1 AS t_f1, s.f1 AS s_f1,
    t.f2 AS t_f2, s.f2 AS s_f2,
    t.f3 AS t_f3, s.f3 AS s_f3,
    t.f4 AS t_f4, s.f4 AS s_f4,
    t.f5 AS t_f5, s.f5 AS s_f5
  FROM workspace.mlpabb808b1.training t
  INNER JOIN workspace.mlpabb808b1.serving s ON t.entity_id = s.entity_id
),
diffs AS (
  SELECT
    entity_id,
    ABS(t_f1 - s_f1) AS diff_f1,
    ABS(t_f2 - s_f2) AS diff_f2,
    ABS(t_f3 - s_f3) AS diff_f3,
    ABS(t_f4 - s_f4) AS diff_f4,
    ABS(t_f5 - s_f5) AS diff_f5
  FROM joined
)
SELECT
  AVG(diff_f1) AS avg_diff_f1,
  AVG(diff_f2) AS avg_diff_f2,
  AVG(diff_f3) AS avg_diff_f3,
  AVG(diff_f4) AS avg_diff_f4,
  AVG(diff_f5) AS avg_diff_f5,
  MAX(diff_f1) AS max_diff_f1,
  MAX(diff_f2) AS max_diff_f2,
  MAX(diff_f3) AS max_diff_f3,
  MAX(diff_f4) AS max_diff_f4,
  MAX(diff_f5) AS max_diff_f5,
  COUNT(*) AS n_common_entities
FROM diffs
""")

print("Results saved to workspace.mlpabb808b1.skew_results!")
