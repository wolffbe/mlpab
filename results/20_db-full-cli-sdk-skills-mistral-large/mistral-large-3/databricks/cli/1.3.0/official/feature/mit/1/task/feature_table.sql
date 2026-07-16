-- Databricks SQL Notebook to create the feature table
-- Read raw data from Volume
CREATE OR REPLACE TABLE workspace.mlpab858a9e.featuresfbc05f AS
WITH transactions AS (
  SELECT
    row_id,
    account_id,
    CAST(event_time AS BIGINT) / 1000 AS event_time_seconds,
    amount,
    currency
  FROM csv."/Volumes/workspace/mlpab858a9e/raw/transactions.csv"
  WITH (
    header => true,
    inferSchema => true
  )
),
fx_rates AS (
  SELECT
    currency,
    fx_rate
  FROM csv."/Volumes/workspace/mlpab858a9e/raw/fx_rates.csv"
  WITH (
    header => true,
    inferSchema => true
  )
),
joined AS (
  SELECT
    t.row_id,
    t.account_id,
    t.event_time_seconds * 1000 AS event_time,
    t.amount,
    t.currency,
    f.fx_rate,
    t.amount * f.fx_rate AS amount_usd,
    CASE WHEN dayofweek(from_unixtime(t.event_time_seconds)) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend
  FROM transactions t
  LEFT JOIN fx_rates f ON t.currency = f.currency
),
windowed AS (
  SELECT
    row_id,
    account_id,
    event_time,
    amount_usd,
    is_weekend,
    SUM(amount) OVER (
      PARTITION BY account_id
      ORDER BY event_time
      RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW
    ) AS amount_7d
  FROM joined
)
SELECT
  row_id,
  account_id,
  event_time,
  amount_usd,
  is_weekend,
  amount_7d
FROM windowed