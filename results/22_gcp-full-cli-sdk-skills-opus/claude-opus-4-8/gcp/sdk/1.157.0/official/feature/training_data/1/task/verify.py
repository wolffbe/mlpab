import os
from google.cloud import bigquery
P = os.environ["GCP_PROJECT"]; DS = os.environ["GCP_BQ_DATASET"]
c = bigquery.Client(project=P); D = f"`{P}.{DS}`"

# No future leakage across ALL feature tables: recompute chosen event_time per feature, compare to label_time
q = f"""
WITH tx AS (
  SELECT account_id,event_time FROM {D}.transactions
  UNION ALL SELECT account_id,event_time FROM {D}.transactions_late),
chk AS (
  SELECT t.account_id,t.label_time,
    (SELECT MAX(event_time) FROM tx x WHERE x.account_id=t.account_id AND x.event_time<=t.label_time) tx_et,
    (SELECT MAX(event_time) FROM {D}.profiles p WHERE p.account_id=t.account_id AND p.event_time<=t.label_time) p_et,
    (SELECT MAX(event_time) FROM {D}.activity a WHERE a.account_id=t.account_id AND a.event_time<=t.label_time) a_et,
    (SELECT MAX(event_time) FROM {D}.account_health h WHERE h.account_id=t.account_id AND h.event_time<=t.label_time) h_et
  FROM {D}.churntrainingcdae59 t)
SELECT COUNTIF(tx_et>label_time OR p_et>label_time OR a_et>label_time OR h_et>label_time) AS leak,
       COUNT(*) AS n FROM chk
"""
print("leak check:", dict(list(c.query(q).result())[0]))

# Confirm every training row's amount matches the MAX-event_time union tx value
q2 = f"""
WITH tx AS (
  SELECT account_id,event_time,amount,balance FROM {D}.transactions
  UNION ALL SELECT account_id,event_time,amount,balance FROM {D}.transactions_late),
expect AS (
  SELECT t.account_id,t.label_time,
    ARRAY_AGG(STRUCT(x.amount,x.balance) ORDER BY x.event_time DESC LIMIT 1)[OFFSET(0)] AS v
  FROM {D}.churntrainingcdae59 t JOIN tx x
    ON x.account_id=t.account_id AND x.event_time<=t.label_time
  GROUP BY t.account_id,t.label_time)
SELECT COUNTIF(t.amount!=e.v.amount OR t.balance!=e.v.balance) AS mismatch, COUNT(*) n
FROM {D}.churntrainingcdae59 t JOIN expect e USING(account_id,label_time)
"""
print("amount/balance match check:", dict(list(c.query(q2).result())[0]))

# distinct label rows vs labels count
q3 = f"SELECT COUNT(*) tot, COUNT(DISTINCT FORMAT('%t|%t',account_id,label_time)) dis FROM {D}.churntrainingcdae59"
print("row/distinct:", dict(list(c.query(q3).result())[0]))
print("labels rows:", list(c.query(f"SELECT COUNT(*) n FROM {D}.labels").result())[0].n)
