SELECT
  t.row_id || ',' || t.account_id || ',' ||
  CAST(t.event_time AS varchar) || ',' ||
  CAST(CAST(t.amount * f.fx_rate AS decimal(24,6)) AS varchar) || ',' ||
  CAST(CASE WHEN day_of_week(from_unixtime(t.event_time/1000, 'UTC')) IN (6,7) THEN 1 ELSE 0 END AS varchar) || ',' ||
  CAST(CAST((SELECT SUM(t2.amount) FROM delta.mlpab348f5a_featurestore.transactions_raw_1 t2
     WHERE t2.account_id = t.account_id
       AND t2.event_time BETWEEN t.event_time - 604800000 AND t.event_time) AS decimal(24,6)) AS varchar) AS csvline
FROM delta.mlpab348f5a_featurestore.transactions_raw_1 t
JOIN delta.mlpab348f5a_featurestore.fx_rates_1 f ON t.currency = f.currency
