SELECT
  l.account_id AS account_id,
  l.label_time AS label_time,
  t.amount AS amount,
  t.balance AS balance,
  p.credit_score AS credit_score,
  p.tier AS tier,
  a.sessions_7d AS sessions_7d,
  h.health_score AS health_score,
  l.churned AS churned
FROM delta.mlpabd97552_featurestore.labels_1 l
LEFT JOIN LATERAL (
  SELECT amount, balance FROM delta.mlpabd97552_featurestore.transactions_1 t
  WHERE t.account_id = l.account_id AND t.event_time <= l.label_time
  ORDER BY t.event_time DESC LIMIT 1
) t ON true
LEFT JOIN LATERAL (
  SELECT credit_score, tier FROM delta.mlpabd97552_featurestore.profiles_1 p
  WHERE p.account_id = l.account_id AND p.event_time <= l.label_time
  ORDER BY p.event_time DESC LIMIT 1
) p ON true
LEFT JOIN LATERAL (
  SELECT sessions_7d FROM delta.mlpabd97552_featurestore.activity_1 a
  WHERE a.account_id = l.account_id AND a.event_time <= l.label_time
  ORDER BY a.event_time DESC LIMIT 1
) a ON true
LEFT JOIN LATERAL (
  SELECT health_score FROM delta.mlpabd97552_featurestore.account_health_1 h
  WHERE h.account_id = l.account_id AND h.event_time <= l.label_time
  ORDER BY h.event_time DESC LIMIT 1
) h ON true
ORDER BY l.account_id, l.label_time
