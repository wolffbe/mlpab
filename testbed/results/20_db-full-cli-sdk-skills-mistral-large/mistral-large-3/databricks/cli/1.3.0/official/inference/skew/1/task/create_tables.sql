CREATE OR REPLACE TABLE workspace.mlpab6c9d6b.training AS
SELECT * FROM csv."`dbfs:/Volumes/workspace/mlpab6c9d6b/skew_volume/training_sample.csv`"
WITH (header = true, inferSchema = true);

CREATE OR REPLACE TABLE workspace.mlpab6c9d6b.serving AS
SELECT * FROM csv."`dbfs:/Volumes/workspace/mlpab6c9d6b/skew_volume/serving_log.csv`"
WITH (header = true, inferSchema = true);

-- Compute summary statistics for each feature in training
CREATE OR REPLACE TABLE workspace.mlpab6c9d6b.training_stats AS
SELECT
  feature,
  AVG(value) AS mean,
  STDDEV(value) AS stddev,
  MIN(value) AS min,
  MAX(value) AS max
FROM (
  SELECT 'f1' AS feature, f1 AS value FROM workspace.mlpab6c9d6b.training
  UNION ALL SELECT 'f2' AS feature, f2 AS value FROM workspace.mlpab6c9d6b.training
  UNION ALL SELECT 'f3' AS feature, f3 AS value FROM workspace.mlpab6c9d6b.training
  UNION ALL SELECT 'f4' AS feature, f4 AS value FROM workspace.mlpab6c9d6b.training
  UNION ALL SELECT 'f5' AS feature, f5 AS value FROM workspace.mlpab6c9d6b.training
)
GROUP BY feature;

-- Compute summary statistics for each feature in serving
CREATE OR REPLACE TABLE workspace.mlpab6c9d6b.serving_stats AS
SELECT
  feature,
  AVG(value) AS mean,
  STDDEV(value) AS stddev,
  MIN(value) AS min,
  MAX(value) AS max
FROM (
  SELECT 'f1' AS feature, f1 AS value FROM workspace.mlpab6c9d6b.serving
  UNION ALL SELECT 'f2' AS feature, f2 AS value FROM workspace.mlpab6c9d6b.serving
  UNION ALL SELECT 'f3' AS feature, f3 AS value FROM workspace.mlpab6c9d6b.serving
  UNION ALL SELECT 'f4' AS feature, f4 AS value FROM workspace.mlpab6c9d6b.serving
  UNION ALL SELECT 'f5' AS feature, f5 AS value FROM workspace.mlpab6c9d6b.serving
)
GROUP BY feature;

-- Join and compare statistics
CREATE OR REPLACE TABLE workspace.mlpab6c9d6b.feature_comparison AS
SELECT
  t.feature,
  t.mean AS training_mean,
  s.mean AS serving_mean,
  ABS(t.mean - s.mean) AS mean_diff,
  t.stddev AS training_stddev,
  s.stddev AS serving_stddev,
  ABS(t.stddev - s.stddev) AS stddev_diff,
  t.min AS training_min,
  s.min AS serving_min,
  t.max AS training_max,
  s.max AS serving_max
FROM workspace.mlpab6c9d6b.training_stats t
JOIN workspace.mlpab6c9d6b.serving_stats s
ON t.feature = s.feature;

-- Retrieve the comparison results
SELECT * FROM workspace.mlpab6c9d6b.feature_comparison ORDER BY mean_diff DESC;