CREATE OR REPLACE TABLE `***REDACTED***.mlpab_mlpab7232d6.churntrainingcdae59_v1` AS
WITH tx AS (
  SELECT account_id, event_time, amount, balance FROM `***REDACTED***.mlpab_mlpab7232d6.raw_transactions`
  UNION ALL
  SELECT account_id, event_time, amount, balance FROM `***REDACTED***.mlpab_mlpab7232d6.raw_transactions_late`
),
lab AS (
  SELECT account_id, label_time, churned FROM `***REDACTED***.mlpab_mlpab7232d6.raw_labels`
),
tx_pit AS (
  SELECT l.account_id, l.label_time, t.amount, t.balance
  FROM lab l JOIN tx t
    ON t.account_id = l.account_id AND t.event_time <= l.label_time
  QUALIFY ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY t.event_time DESC) = 1
),
prof_pit AS (
  SELECT l.account_id, l.label_time, p.credit_score, p.tier
  FROM lab l JOIN `***REDACTED***.mlpab_mlpab7232d6.raw_profiles` p
    ON p.account_id = l.account_id AND p.event_time <= l.label_time
  QUALIFY ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY p.event_time DESC) = 1
),
act_pit AS (
  SELECT l.account_id, l.label_time, a.sessions_7d
  FROM lab l JOIN `***REDACTED***.mlpab_mlpab7232d6.raw_activity` a
    ON a.account_id = l.account_id AND a.event_time <= l.label_time
  QUALIFY ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY a.event_time DESC) = 1
),
health_pit AS (
  SELECT l.account_id, l.label_time, h.health_score
  FROM lab l JOIN `***REDACTED***.mlpab_mlpab7232d6.raw_account_health` h
    ON h.account_id = l.account_id AND h.event_time <= l.label_time
  QUALIFY ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY h.event_time DESC) = 1
)
SELECT
  l.account_id,
  l.label_time,
  tx_pit.amount,
  tx_pit.balance,
  prof_pit.credit_score,
  prof_pit.tier,
  act_pit.sessions_7d,
  health_pit.health_score,
  l.churned
FROM lab l
LEFT JOIN tx_pit     USING (account_id, label_time)
LEFT JOIN prof_pit   USING (account_id, label_time)
LEFT JOIN act_pit    USING (account_id, label_time)
LEFT JOIN health_pit USING (account_id, label_time)
