CREATE OR REPLACE TABLE `***REDACTED***.mlpab_mlpab5d6311.cctxn76ccb2` AS
WITH base AS (
  SELECT t.*, cs.card_avg_amount, cs.card_std_amount, cs.card_avg_lat, cs.card_avg_long
  FROM `***REDACTED***.mlpab_mlpab5d6311.raw_txn` t
  JOIN `***REDACTED***.mlpab_mlpab5d6311.card_stats` cs USING(cc_num)
)
SELECT
  transaction_id, cc_num, datetime, amount, merchant, category, lat, long, is_fraud,
  EXTRACT(HOUR FROM datetime) AS hour,
  EXTRACT(DAYOFWEEK FROM datetime) AS dow,
  LOG(amount + 1) AS log_amount,
  SAFE_DIVIDE(amount, card_avg_amount) AS amount_vs_avg,
  SAFE_DIVIDE(amount - card_avg_amount, card_std_amount) AS amount_z,
  ST_DISTANCE(ST_GEOGPOINT(long, lat), ST_GEOGPOINT(card_avg_long, card_avg_lat)) AS geo_dist,
  COUNT(*) OVER (PARTITION BY cc_num ORDER BY UNIX_SECONDS(datetime) RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW) AS velocity_1h,
  COUNT(*) OVER (PARTITION BY cc_num ORDER BY UNIX_SECONDS(datetime) RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW) AS velocity_24h,
  IFNULL(TIMESTAMP_DIFF(datetime, LAG(datetime) OVER(PARTITION BY cc_num ORDER BY datetime), SECOND), 999999) AS secs_since_prev
FROM base
