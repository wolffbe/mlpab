-- Compute scores for all accounts as of T = 1773234000000
-- Input: feature_history.csv in the volume
-- Output: Delta table scores4a1a3b, version 1

-- Step 1: Create a temporary view for the feature history
CREATE OR REPLACE TEMPORARY VIEW feature_history AS
SELECT *
FROM csv.
  "`dbfs:/Volumes/workspace/mlpabbed188/scores4a1a3b_volume/feature_history.csv`"
  (header => true, inferSchema => true);

-- Step 2: Filter to the most recent revision at or before T
CREATE OR REPLACE TEMPORARY VIEW latest_features AS
SELECT
  account_id,
  f1,
  f2,
  f3
FROM (
  SELECT
    account_id,
    f1,
    f2,
    f3,
    event_time,
    ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
  FROM feature_history
  WHERE event_time <= 1773234000000
)
WHERE rn = 1;

-- Step 3: Load the model weights and bias
CREATE OR REPLACE TEMPORARY VIEW model AS
SELECT *
FROM json.
  "`dbfs:/Volumes/workspace/mlpabbed188/scores4a1a3b_volume/model.json`";

-- Step 4: Compute the score
CREATE OR REPLACE TEMPORARY VIEW scores AS
SELECT
  account_id,
  ROUND(
    1.0 / (1.0 + EXP(-(
      (f1 * (SELECT weights.f1 FROM model)) +
      (f2 * (SELECT weights.f2 FROM model)) +
      (f3 * (SELECT weights.f3 FROM model)) +
      (SELECT bias FROM model)
    ))),
    6
  ) as score
FROM latest_features;

-- Step 5: Write the results to a Delta table
CREATE OR REPLACE TABLE `workspace`.`mlpabbed188`.`scores4a1a3b` AS
SELECT * FROM scores;

-- Step 6: Enable online access for low-latency lookup
CREATE OR REPLACE ONLINE TABLE `workspace`.`mlpabbed188`.`scores4a1a3b_online`
AS SELECT * FROM `workspace`.`mlpabbed188`.`scores4a1a3b`;