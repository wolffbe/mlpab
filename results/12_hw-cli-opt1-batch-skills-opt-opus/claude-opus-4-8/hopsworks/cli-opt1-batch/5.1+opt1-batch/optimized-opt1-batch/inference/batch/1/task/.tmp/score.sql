WITH ranked AS (
  SELECT account_id, f1, f2, f3,
         row_number() OVER (PARTITION BY account_id ORDER BY event_time DESC) AS rn
  FROM delta.mlpab6690f4_featurestore.feature_history_1
  WHERE event_time <= 1773594000000
)
SELECT account_id,
       round(1.0 / (1.0 + exp(-(-0.7952*f1 + 0.3018*f2 - 0.8863*f3 - 0.2124))), 6) AS score
FROM ranked
WHERE rn = 1
ORDER BY account_id
