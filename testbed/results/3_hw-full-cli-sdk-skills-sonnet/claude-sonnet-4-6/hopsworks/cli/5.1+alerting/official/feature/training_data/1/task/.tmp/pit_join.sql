WITH labels AS (
  SELECT account_id, label_time, churned
  FROM delta.mlpaba11756_featurestore.churn_labels_1
),
tx_ranked AS (
  SELECT l.account_id, l.label_time, t.amount, t.balance,
         ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY t.event_time DESC) AS rn
  FROM labels l
  LEFT JOIN delta.mlpaba11756_featurestore.churn_transactions_1 t
    ON t.account_id = l.account_id AND t.event_time <= l.label_time
),
tx AS (
  SELECT account_id, label_time, amount, balance FROM tx_ranked WHERE rn = 1
),
pr_ranked AS (
  SELECT l.account_id, l.label_time, p.credit_score, p.tier,
         ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY p.event_time DESC) AS rn
  FROM labels l
  LEFT JOIN delta.mlpaba11756_featurestore.churn_profiles_1 p
    ON p.account_id = l.account_id AND p.event_time <= l.label_time
),
pr AS (
  SELECT account_id, label_time, credit_score, tier FROM pr_ranked WHERE rn = 1
),
ac_ranked AS (
  SELECT l.account_id, l.label_time, a.sessions_7d,
         ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY a.event_time DESC) AS rn
  FROM labels l
  LEFT JOIN delta.mlpaba11756_featurestore.churn_activity_1 a
    ON a.account_id = l.account_id AND a.event_time <= l.label_time
),
ac AS (
  SELECT account_id, label_time, sessions_7d FROM ac_ranked WHERE rn = 1
),
h_ranked AS (
  SELECT l.account_id, l.label_time, h.health_score,
         ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY h.event_time DESC) AS rn
  FROM labels l
  LEFT JOIN delta.mlpaba11756_featurestore.churn_health_1 h
    ON h.account_id = l.account_id AND h.event_time <= l.label_time
),
h AS (
  SELECT account_id, label_time, health_score FROM h_ranked WHERE rn = 1
)
SELECT
  l.account_id,
  l.label_time,
  tx.amount,
  tx.balance,
  pr.credit_score,
  pr.tier,
  ac.sessions_7d,
  h.health_score,
  l.churned
FROM labels l
LEFT JOIN tx ON tx.account_id = l.account_id AND tx.label_time = l.label_time
LEFT JOIN pr ON pr.account_id = l.account_id AND pr.label_time = l.label_time
LEFT JOIN ac ON ac.account_id = l.account_id AND ac.label_time = l.label_time
LEFT JOIN h ON h.account_id = l.account_id AND h.label_time = l.label_time
ORDER BY l.account_id
