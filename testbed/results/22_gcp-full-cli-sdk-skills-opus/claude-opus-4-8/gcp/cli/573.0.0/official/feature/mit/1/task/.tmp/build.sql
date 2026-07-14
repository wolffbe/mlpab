SELECT
  t.row_id,
  t.account_id,
  t.event_time,
  t.amount * f.fx_rate AS amount_usd,
  IF(EXTRACT(DAYOFWEEK FROM TIMESTAMP_MILLIS(t.event_time)) IN (1, 7), 1, 0) AS is_weekend,
  (SELECT SUM(t2.amount)
     FROM `***REDACTED***.mlpab_mlpab3ff88e.stg_transactions` t2
    WHERE t2.account_id = t.account_id
      AND t2.event_time BETWEEN t.event_time - 604800000 AND t.event_time
  ) AS amount_7d
FROM `***REDACTED***.mlpab_mlpab3ff88e.stg_transactions` t
JOIN `***REDACTED***.mlpab_mlpab3ff88e.stg_fx_rates` f USING (currency)
