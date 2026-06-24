SELECT
  t.row_id,
  t.account_id,
  t.event_time,
  t.amount * f.fx_rate AS amount_usd,
  CASE WHEN (CAST(floor(t.event_time / 86400000) AS bigint) % 7) IN (2, 3) THEN 1 ELSE 0 END AS is_weekend,
  SUM(t.amount) OVER (
    PARTITION BY t.account_id
    ORDER BY t.event_time
    RANGE BETWEEN 604800000 PRECEDING AND CURRENT ROW
  ) AS amount_7d
FROM delta.mlpabffb061_featurestore.transactions_raw_1 t
JOIN delta.mlpabffb061_featurestore.fx_rates_1 f ON t.currency = f.currency
ORDER BY t.account_id, t.event_time
