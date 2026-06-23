WITH
tx AS (
  SELECT account_id, label_time, amount, balance FROM (
    SELECT l.account_id, l.label_time, t.amount, t.balance,
      row_number() OVER (PARTITION BY l.account_id, l.label_time ORDER BY t.event_time DESC) rn
    FROM delta.mlpabd97552_featurestore.labels_1 l
    JOIN delta.mlpabd97552_featurestore.transactions_1 t
      ON t.account_id = l.account_id AND t.event_time <= l.label_time
  ) WHERE rn = 1
),
pr AS (
  SELECT account_id, label_time, credit_score, tier FROM (
    SELECT l.account_id, l.label_time, p.credit_score, p.tier,
      row_number() OVER (PARTITION BY l.account_id, l.label_time ORDER BY p.event_time DESC) rn
    FROM delta.mlpabd97552_featurestore.labels_1 l
    JOIN delta.mlpabd97552_featurestore.profiles_1 p
      ON p.account_id = l.account_id AND p.event_time <= l.label_time
  ) WHERE rn = 1
),
ac AS (
  SELECT account_id, label_time, sessions_7d FROM (
    SELECT l.account_id, l.label_time, a.sessions_7d,
      row_number() OVER (PARTITION BY l.account_id, l.label_time ORDER BY a.event_time DESC) rn
    FROM delta.mlpabd97552_featurestore.labels_1 l
    JOIN delta.mlpabd97552_featurestore.activity_1 a
      ON a.account_id = l.account_id AND a.event_time <= l.label_time
  ) WHERE rn = 1
),
ah AS (
  SELECT account_id, label_time, health_score FROM (
    SELECT l.account_id, l.label_time, h.health_score,
      row_number() OVER (PARTITION BY l.account_id, l.label_time ORDER BY h.event_time DESC) rn
    FROM delta.mlpabd97552_featurestore.labels_1 l
    JOIN delta.mlpabd97552_featurestore.account_health_1 h
      ON h.account_id = l.account_id AND h.event_time <= l.label_time
  ) WHERE rn = 1
)
SELECT
  l.account_id AS account_id,
  l.label_time AS label_time,
  tx.amount AS amount,
  tx.balance AS balance,
  pr.credit_score AS credit_score,
  pr.tier AS tier,
  ac.sessions_7d AS sessions_7d,
  ah.health_score AS health_score,
  l.churned AS churned
FROM delta.mlpabd97552_featurestore.labels_1 l
LEFT JOIN tx ON tx.account_id = l.account_id AND tx.label_time = l.label_time
LEFT JOIN pr ON pr.account_id = l.account_id AND pr.label_time = l.label_time
LEFT JOIN ac ON ac.account_id = l.account_id AND ac.label_time = l.label_time
LEFT JOIN ah ON ah.account_id = l.account_id AND ah.label_time = l.label_time
ORDER BY l.account_id, l.label_time
