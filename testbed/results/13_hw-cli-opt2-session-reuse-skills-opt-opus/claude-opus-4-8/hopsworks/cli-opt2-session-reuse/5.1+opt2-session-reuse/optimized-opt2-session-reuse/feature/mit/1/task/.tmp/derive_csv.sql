SELECT
  t.row_id || ',' || t.account_id || ',' ||
  CAST(t.event_time AS varchar) || ',' ||
  CAST(t.amount * f.fx_rate AS varchar) || ',' ||
  CAST(CASE WHEN day_of_week(from_unixtime(t.event_time / 1000, 'UTC')) IN (6, 7) THEN 1 ELSE 0 END AS varchar) || ',' ||
  CAST(SUM(t.amount) OVER (
    PARTITION BY t.account_id
    ORDER BY t.event_time
    RANGE BETWEEN 604800000 PRECEDING AND CURRENT ROW
  ) AS varchar) AS line
FROM transactions_raw56bbbb_1 t
JOIN fx_rates56bbbb_1 f ON t.currency = f.currency
ORDER BY t.row_id
