SELECT
  COUNT(*) AS n,
  COUNT(DISTINCT row_id) AS distinct_keys,
  COUNTIF(amount_usd IS NULL) AS usd_null,
  COUNTIF(amount_7d IS NULL) AS w_null,
  SUM(is_weekend) AS weekend_rows
FROM `***REDACTED***.mlpab_mlpab3ff88e.features347afc`
