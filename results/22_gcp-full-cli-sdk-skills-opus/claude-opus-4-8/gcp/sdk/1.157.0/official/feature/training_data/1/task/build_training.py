import os
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
client = bigquery.Client(project=PROJECT)
D = f"`{PROJECT}.{DATASET}`"
OUT = f"{PROJECT}.{DATASET}.churntrainingcdae59"

sql = f"""
CREATE OR REPLACE TABLE `{OUT}` AS
WITH
labels AS (
  SELECT account_id, label_time, churned,
         ROW_NUMBER() OVER (ORDER BY account_id, label_time) AS rid
  FROM {D}.labels
),
tx AS (
  SELECT account_id, event_time, amount, balance FROM {D}.transactions
  UNION ALL
  SELECT account_id, event_time, amount, balance FROM {D}.transactions_late
),
tx_pit AS (
  SELECT l.rid, t.amount, t.balance
  FROM labels l JOIN tx t
    ON t.account_id = l.account_id AND t.event_time <= l.label_time
  QUALIFY ROW_NUMBER() OVER (PARTITION BY l.rid ORDER BY t.event_time DESC) = 1
),
prof_pit AS (
  SELECT l.rid, p.credit_score, p.tier
  FROM labels l JOIN {D}.profiles p
    ON p.account_id = l.account_id AND p.event_time <= l.label_time
  QUALIFY ROW_NUMBER() OVER (PARTITION BY l.rid ORDER BY p.event_time DESC) = 1
),
act_pit AS (
  SELECT l.rid, a.sessions_7d
  FROM labels l JOIN {D}.activity a
    ON a.account_id = l.account_id AND a.event_time <= l.label_time
  QUALIFY ROW_NUMBER() OVER (PARTITION BY l.rid ORDER BY a.event_time DESC) = 1
),
health_pit AS (
  SELECT l.rid, h.health_score
  FROM labels l JOIN {D}.account_health h
    ON h.account_id = l.account_id AND h.event_time <= l.label_time
  QUALIFY ROW_NUMBER() OVER (PARTITION BY l.rid ORDER BY h.event_time DESC) = 1
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
FROM labels l
LEFT JOIN tx_pit     USING (rid)
LEFT JOIN prof_pit   USING (rid)
LEFT JOIN act_pit    USING (rid)
LEFT JOIN health_pit USING (rid)
ORDER BY l.account_id, l.label_time
"""

client.query(sql).result()

tbl = client.get_table(OUT)
print("columns:", [f.name for f in tbl.schema])
print("rows:", tbl.num_rows)

# sanity preview + null counts
rows = list(client.query(
    f"SELECT * FROM `{OUT}` LIMIT 5").result())
for r in rows:
    print(dict(r))

nulls = list(client.query(f"""
SELECT
  COUNTIF(amount IS NULL) amount_n,
  COUNTIF(balance IS NULL) balance_n,
  COUNTIF(credit_score IS NULL) credit_n,
  COUNTIF(tier IS NULL) tier_n,
  COUNTIF(sessions_7d IS NULL) sess_n,
  COUNTIF(health_score IS NULL) health_n,
  COUNT(*) total
FROM `{OUT}`""").result())[0]
print("null counts:", dict(nulls))
print("DONE")
