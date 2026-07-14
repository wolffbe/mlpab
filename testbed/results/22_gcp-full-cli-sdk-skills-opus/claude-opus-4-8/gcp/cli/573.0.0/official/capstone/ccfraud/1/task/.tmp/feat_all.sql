CREATE OR REPLACE TABLE `mlpab_mlpab08f695.feat_all` AS
WITH combined AS (
  SELECT transaction_id, cc_num, datetime, amount, merchant, category, lat, long,
         CAST(is_fraud AS INT64) AS is_fraud, 'train' AS src
  FROM `mlpab_mlpab08f695.raw_transactions`
  UNION ALL
  SELECT transaction_id, cc_num, datetime, amount, merchant, category, lat, long,
         CAST(NULL AS INT64) AS is_fraud, 'score' AS src
  FROM `mlpab_mlpab08f695.raw_score`
),
profile AS (
  SELECT cc_num,
         AVG(lat) AS home_lat, AVG(long) AS home_long,
         AVG(amount) AS avg_amt, STDDEV(amount) AS std_amt,
         COUNT(*) AS card_txn_cnt
  FROM combined
  GROUP BY cc_num
),
base AS (
  SELECT c.*, p.home_lat, p.home_long, p.avg_amt, p.std_amt, p.card_txn_cnt,
         UNIX_SECONDS(c.datetime) AS ts
  FROM combined c JOIN profile p USING (cc_num)
)
SELECT
  transaction_id, cc_num, datetime, src, is_fraud,
  amount,
  category,
  LOG10(amount + 1) AS log_amt,
  EXTRACT(HOUR FROM datetime) AS hour,
  EXTRACT(DAYOFWEEK FROM datetime) AS dow,
  IF(EXTRACT(HOUR FROM datetime) < 6, 1, 0) AS is_night,
  -- haversine distance (km) from card home location
  2 * 6371 * ASIN(SQRT(
      POW(SIN(( (lat - home_lat) * ACOS(-1)/180 )/2), 2) +
      COS(lat*ACOS(-1)/180) * COS(home_lat*ACOS(-1)/180) *
      POW(SIN(( (long - home_long) * ACOS(-1)/180 )/2), 2)
  )) AS geo_dist_km,
  amount / NULLIF(avg_amt, 0) AS amt_over_avg,
  (amount - avg_amt) / NULLIF(std_amt, 0) AS amt_z,
  card_txn_cnt,
  COUNT(*) OVER (PARTITION BY cc_num ORDER BY ts RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING) AS velocity_1h,
  COUNT(*) OVER (PARTITION BY cc_num ORDER BY ts RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING) AS velocity_24h,
  COALESCE(ts - LAG(ts) OVER (PARTITION BY cc_num ORDER BY ts), 999999) AS secs_since_prev,
  CURRENT_TIMESTAMP() AS feature_timestamp
FROM base
