SELECT concat(
  t.row_id, ',',
  t.account_id, ',',
  cast(t.event_time as varchar), ',',
  cast(t.amount * f.fx_rate as varchar), ',',
  cast(CASE WHEN (CAST(floor(t.event_time / 86400000) AS bigint) % 7) IN (2, 3) THEN 1 ELSE 0 END as varchar), ',',
  cast(SUM(t.amount) OVER (
    PARTITION BY t.account_id
    ORDER BY t.event_time
    RANGE BETWEEN 604800000 PRECEDING AND CURRENT ROW
  ) as varchar)
) AS csv
FROM delta.mlpabffb061_featurestore.transactions_raw_1 t
JOIN delta.mlpabffb061_featurestore.fx_rates_1 f ON t.currency = f.currency
ORDER BY t.account_id, t.event_time
