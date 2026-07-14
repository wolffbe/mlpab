-- Databricks SQL Notebook
-- This notebook processes the feature history and computes scores.

-- Load inputs
CREATE OR REPLACE TEMP VIEW feature_history AS
SELECT * FROM csv."`/Volumes/workspace/mlpab203bd0/mlpab203bd0_volume/feature_history.csv`"
WITH (
  header = "true",
  inferSchema = "true"
);

-- Load model
CREATE OR REPLACE TEMP VIEW model AS
SELECT from_json(value, 'STRUCT<weights: STRUCT<f1: DOUBLE, f2: DOUBLE, f3: DOUBLE>, bias: DOUBLE>') AS model
FROM text."`/Volumes/workspace/mlpab203bd0/mlpab203bd0_volume/model.json`";

-- Filter feature history to retain only the most recent revision at or before T
CREATE OR REPLACE TEMP VIEW filtered AS
WITH ranked AS (
  SELECT 
    account_id,
    event_time,
    f1, f2, f3,
    ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) AS rank
  FROM feature_history
  WHERE event_time <= 1773234000000
)
SELECT account_id, f1, f2, f3
FROM ranked
WHERE rank = 1;

-- Compute scores
CREATE OR REPLACE TABLE workspace.mlpab203bd0.scores4a1a3b AS
WITH weights AS (
  SELECT 
    model.weights.f1 AS w_f1,
    model.weights.f2 AS w_f2,
    model.weights.f3 AS w_f3,
    model.bias AS bias
  FROM model
)
SELECT 
  account_id,
  ROUND(1 / (1 + EXP(-(
    w_f1 * f1 + 
    w_f2 * f2 + 
    w_f3 * f3 + 
    bias
  ))), 6) AS score
FROM filtered, weights;

-- Enable online access for low-latency lookup
CREATE TABLE IF NOT EXISTS workspace.mlpab203bd0.scores4a1a3b ONLINE VERSION 1;