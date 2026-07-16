import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout, StatementState

w = WorkspaceClient()
S = "workspace.mlpab67db84"
VOL = "/Volumes/workspace/mlpab67db84/raw"

WH = None
for wh in w.warehouses.list():
    if "grader" in (wh.name or ""):
        WH = wh.id
if WH is None:
    WH = next(iter(w.warehouses.list())).id
print("warehouse:", WH)


def ensure_running(max_min=45):
    from databricks.sdk.service.sql import State
    deadline = time.time() + max_min * 60
    while time.time() < deadline:
        try:
            wh = w.warehouses.get(WH)
        except Exception as e:
            print("wh get retry:", str(e)[:150], flush=True)
            time.sleep(20)
            continue
        if wh.state == State.RUNNING:
            print("warehouse RUNNING", flush=True)
            return
        if wh.state in (State.STOPPED, State.DELETED):
            try:
                w.warehouses.start(WH)
                print("issued start", flush=True)
            except Exception as e:
                print("start err:", str(e)[:200], flush=True)
        time.sleep(30)
    raise TimeoutError("warehouse never reached RUNNING")


def run_once(stmt, timeout_min):
    st = None
    for _ in range(5):
        try:
            st = w.statement_execution.execute_statement(
                warehouse_id=WH,
                statement=stmt,
                wait_timeout="30s",
                on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
            )
            break
        except Exception as e:
            print("submit retry:", type(e).__name__, str(e)[:200], flush=True)
            time.sleep(15)
    if st is None:
        raise RuntimeError("could not submit statement")
    deadline = time.time() + timeout_min * 60
    while st.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if time.time() > deadline:
            raise TimeoutError(stmt[:120])
        time.sleep(5)
        try:
            st = w.statement_execution.get_statement(st.statement_id)
        except Exception as e:
            print("poll retry:", type(e).__name__, str(e)[:200], flush=True)
            time.sleep(15)
    return st


def sql(stmt, timeout_min=20):
    for attempt in range(6):
        st = run_once(stmt, timeout_min)
        if st.status.state == StatementState.SUCCEEDED:
            return st.result.data_array if st.result and st.result.data_array else []
        msg = str(st.status.error)
        print(f"statement {st.status.state}: {msg[:200]}", flush=True)
        if "stopped" in msg or "RESOURCE_EXHAUSTED" in msg or "cluster" in msg.lower():
            ensure_running()
            continue
        raise RuntimeError(f"SQL failed [{st.status.state}]: {st.status.error}\n--- statement:\n{stmt[:2000]}")
    raise RuntimeError("statement kept failing after warehouse restarts")


ensure_running()


HAV = (
    "2*6371*ASIN(SQRT(LEAST(1.0,"
    " POWER(SIN(RADIANS({lat2}-{lat1})/2),2)"
    " + COS(RADIANS({lat1}))*COS(RADIANS({lat2}))*POWER(SIN(RADIANS({lon2}-{lon1})/2),2))))"
)

# ---------- raw tables ----------
sql(f"""CREATE OR REPLACE TABLE {S}.cc_raw_train AS
SELECT CAST(transaction_id AS STRING) transaction_id, CAST(cc_num AS BIGINT) cc_num,
       CAST(datetime AS TIMESTAMP) datetime, CAST(amount AS DOUBLE) amount,
       CAST(merchant AS STRING) merchant, CAST(category AS STRING) category,
       CAST(lat AS DOUBLE) lat, CAST(`long` AS DOUBLE) `long`, CAST(is_fraud AS INT) is_fraud
FROM read_files('{VOL}/transactions.csv', format => 'csv', header => true)""")
sql(f"""CREATE OR REPLACE TABLE {S}.cc_raw_score AS
SELECT CAST(transaction_id AS STRING) transaction_id, CAST(cc_num AS BIGINT) cc_num,
       CAST(datetime AS TIMESTAMP) datetime, CAST(amount AS DOUBLE) amount,
       CAST(merchant AS STRING) merchant, CAST(category AS STRING) category,
       CAST(lat AS DOUBLE) lat, CAST(`long` AS DOUBLE) `long`
FROM read_files('{VOL}/score_transactions.csv', format => 'csv', header => true)""")
print("raw:", sql(f"SELECT (SELECT COUNT(*) FROM {S}.cc_raw_train), (SELECT COUNT(*) FROM {S}.cc_raw_score)"))

# ---------- feature group cctxn9d5953 ----------
dist_prev = HAV.format(lat1="prev_lat", lon2="`long`", lat2="lat", lon1="prev_long")
dist_home = HAV.format(lat1="home_lat", lon2="`long`", lat2="lat", lon1="home_long")
sql(f"""CREATE OR REPLACE TABLE {S}.cctxn9d5953
TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
WITH allt AS (
  SELECT transaction_id, cc_num, datetime, amount, merchant, category, lat, `long`, is_fraud, 1 AS is_train
  FROM {S}.cc_raw_train
  UNION ALL
  SELECT transaction_id, cc_num, datetime, amount, merchant, category, lat, `long`, CAST(NULL AS INT), 0
  FROM {S}.cc_raw_score
),
seq AS (
  SELECT *,
    unix_timestamp(datetime) AS ts,
    LAG(unix_timestamp(datetime)) OVER (PARTITION BY cc_num ORDER BY datetime, transaction_id) AS prev_ts,
    LAG(lat)    OVER (PARTITION BY cc_num ORDER BY datetime, transaction_id) AS prev_lat,
    LAG(`long`) OVER (PARTITION BY cc_num ORDER BY datetime, transaction_id) AS prev_long,
    COUNT(*) OVER (PARTITION BY cc_num ORDER BY unix_timestamp(datetime)
                   RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW) AS cnt_1h,
    COUNT(*) OVER (PARTITION BY cc_num ORDER BY unix_timestamp(datetime)
                   RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW) AS cnt_24h
  FROM allt
),
prof AS (
  SELECT cc_num, AVG(amount) card_amt_mean, STDDEV(amount) card_amt_std,
         PERCENTILE(lat, 0.5) home_lat, PERCENTILE(`long`, 0.5) home_long,
         CAST(COUNT(*) AS DOUBLE) card_n
  FROM {S}.cc_raw_train GROUP BY cc_num
),
glob AS (SELECT AVG(CAST(is_fraud AS DOUBLE)) g FROM {S}.cc_raw_train),
catte AS (
  SELECT category, (SUM(is_fraud) + 20*MAX(g)) / (COUNT(*) + 20) AS cat_te
  FROM {S}.cc_raw_train CROSS JOIN glob GROUP BY category
),
merte AS (
  SELECT merchant, (SUM(is_fraud) + 10*MAX(g)) / (COUNT(*) + 10) AS mer_te
  FROM {S}.cc_raw_train CROSS JOIN glob GROUP BY merchant
)
SELECT
  s.transaction_id, s.cc_num, s.datetime, s.amount,
  LN(1.0 + s.amount) AS log_amount,
  CAST(hour(s.datetime) AS DOUBLE) AS hour,
  CAST(dayofweek(s.datetime) AS DOUBLE) AS dayofweek,
  CASE WHEN hour(s.datetime) BETWEEN 0 AND 5 THEN 1.0 ELSE 0.0 END AS is_night,
  CAST(s.ts - s.prev_ts AS DOUBLE) AS secs_since_prev,
  CASE WHEN s.prev_lat IS NULL THEN CAST(NULL AS DOUBLE) ELSE {dist_prev} END AS dist_prev_km,
  CASE WHEN s.prev_ts IS NULL OR s.ts = s.prev_ts THEN CAST(NULL AS DOUBLE)
       ELSE ({dist_prev}) / ((s.ts - s.prev_ts) / 3600.0) END AS speed_kmh,
  CAST(s.cnt_1h AS DOUBLE) AS cnt_1h,
  CAST(s.cnt_24h AS DOUBLE) AS cnt_24h,
  p.card_amt_mean, p.card_amt_std, p.card_n,
  (s.amount - p.card_amt_mean) / (COALESCE(p.card_amt_std, 0.0) + 1.0) AS amt_z,
  CASE WHEN p.home_lat IS NULL THEN CAST(NULL AS DOUBLE) ELSE {dist_home} END AS dist_home_km,
  COALESCE(c.cat_te, (SELECT g FROM glob)) AS cat_te,
  COALESCE(m.mer_te, (SELECT g FROM glob)) AS mer_te,
  s.is_fraud, s.is_train
FROM seq s
LEFT JOIN prof p ON s.cc_num = p.cc_num
LEFT JOIN catte c ON s.category = c.category
LEFT JOIN merte m ON s.merchant = m.merchant""")
sql(f"ALTER TABLE {S}.cctxn9d5953 ALTER COLUMN transaction_id SET NOT NULL")
sql(f"ALTER TABLE {S}.cctxn9d5953 ADD CONSTRAINT cctxn9d5953_pk PRIMARY KEY(transaction_id)")
print("feature group rows:", sql(f"SELECT COUNT(*) FROM {S}.cctxn9d5953"))

# ---------- training dataset cctd9d5953 ----------
FEATURES = [
    "amount", "log_amount", "hour", "dayofweek", "is_night", "secs_since_prev",
    "dist_prev_km", "speed_kmh", "cnt_1h", "cnt_24h", "card_amt_mean",
    "card_amt_std", "card_n", "amt_z", "dist_home_km", "cat_te", "mer_te",
]
sql(f"""CREATE OR REPLACE TABLE {S}.cctd9d5953 AS
SELECT transaction_id, datetime, {', '.join(FEATURES)}, is_fraud
FROM {S}.cctxn9d5953 WHERE is_train = 1""")
print("training dataset rows:", sql(f"SELECT COUNT(*), AVG(CAST(is_fraud AS DOUBLE)) FROM {S}.cctd9d5953"))

# ---------- long format for the binned log-odds (naive bayes) model ----------
MODEL_FEATS = [
    "log_amount", "amt_z", "dist_home_km", "dist_prev_km", "speed_kmh",
    "secs_since_prev", "cnt_1h", "cnt_24h", "cat_te", "mer_te", "card_n",
    "hour", "dayofweek",
]
CATEG = ("hour", "dayofweek")
stack_args = ", ".join(f"'{f}', CAST({f} AS DOUBLE)" for f in MODEL_FEATS)
sql(f"""CREATE OR REPLACE TABLE {S}.cc_feat_long AS
SELECT transaction_id, is_train, is_fraud, unix_timestamp(datetime) AS ts,
       stack({len(MODEL_FEATS)}, {stack_args}) AS (fname, fval)
FROM {S}.cctxn9d5953""")

t_split = int(float(sql(f"SELECT CAST(percentile(unix_timestamp(datetime), 0.8) AS BIGINT) FROM {S}.cctd9d5953")[0][0]))
print("time split:", t_split)


def fit(tag, train_filter):
    categ = "', '".join(CATEG)
    sql(f"""CREATE OR REPLACE TABLE {S}.cc_bins_{tag} AS
SELECT fname, approx_percentile(fval, array(0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95)) AS edges
FROM {S}.cc_feat_long
WHERE is_train = 1 AND ({train_filter}) AND fval IS NOT NULL AND fname NOT IN ('{categ}')
GROUP BY fname""")
    sql(f"""CREATE OR REPLACE TABLE {S}.cc_binned_{tag} AS
SELECT f.transaction_id, f.is_train, f.is_fraud, f.ts, f.fname,
  CASE WHEN f.fname IN ('{categ}') THEN CAST(f.fval AS INT)
       WHEN f.fval IS NULL THEN -1
       ELSE size(filter(b.edges, e -> f.fval > e)) END AS bin
FROM {S}.cc_feat_long f
LEFT JOIN {S}.cc_bins_{tag} b ON f.fname = b.fname""")
    sql(f"""CREATE OR REPLACE TABLE {S}.cc_weights_{tag} AS
WITH tot AS (
  SELECT SUM(is_fraud) pos, SUM(1 - is_fraud) neg
  FROM {S}.cc_binned_{tag} WHERE fname = 'log_amount' AND is_train = 1 AND ({train_filter})
),
agg AS (
  SELECT fname, bin, SUM(is_fraud) p, SUM(1 - is_fraud) n
  FROM {S}.cc_binned_{tag} WHERE is_train = 1 AND ({train_filter})
  GROUP BY fname, bin
)
SELECT fname, bin,
  LN((p + 1.0) / (tot.pos + 24.0)) - LN((n + 1.0) / (tot.neg + 24.0)) AS w
FROM agg CROSS JOIN tot""")
    sql(f"""CREATE OR REPLACE TABLE {S}.cc_scored_{tag} AS
SELECT b.transaction_id, MAX(b.is_train) is_train, MAX(b.is_fraud) is_fraud, MAX(b.ts) ts,
       SUM(COALESCE(w.w, 0.0)) AS score
FROM {S}.cc_binned_{tag} b
LEFT JOIN {S}.cc_weights_{tag} w ON b.fname = w.fname AND b.bin = w.bin
GROUP BY b.transaction_id""")


def auc(where):
    rows = sql(f"""WITH e AS (
  SELECT score, CAST(is_fraud AS DOUBLE) y FROM {S}.cc_scored_cv WHERE {where}
),
rk AS (
  SELECT y, AVG(rn) OVER (PARTITION BY score) ar
  FROM (SELECT score, y, CAST(ROW_NUMBER() OVER (ORDER BY score) AS DOUBLE) rn FROM e)
),
a AS (SELECT SUM(CASE WHEN y = 1 THEN ar ELSE 0 END) sp, SUM(y) np, SUM(1 - y) nn FROM rk)
SELECT (sp - np * (np + 1) / 2) / (np * nn) FROM a""")
    return float(rows[0][0])


fit("cv", f"ts <= {t_split}")
holdout_auc = auc(f"is_train = 1 AND ts > {t_split}")
train_auc = auc(f"is_train = 1 AND ts <= {t_split}")
print("holdout ROC AUC:", holdout_auc, " (train-slice AUC:", train_auc, ")")

# final model on all labelled data
fit("full", "TRUE")

# ---------- predictions ccpred9d5953 ----------
sql(f"""CREATE OR REPLACE TABLE {S}.ccpred9d5953 (
  transaction_id STRING NOT NULL,
  fraud_probability DOUBLE,
  CONSTRAINT ccpred9d5953_pk PRIMARY KEY (transaction_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)""")
sql(f"""INSERT INTO {S}.ccpred9d5953
SELECT transaction_id,
       LEAST(1.0, GREATEST(0.0,
         1.0 / (1.0 + EXP(-(score + (SELECT LN(SUM(is_fraud) / SUM(1 - is_fraud)) FROM {S}.cctd9d5953))))
       )) AS fraud_probability
FROM {S}.cc_scored_full WHERE is_train = 0""")
stats = sql(f"SELECT COUNT(*), MIN(fraud_probability), MAX(fraud_probability), AVG(fraud_probability) FROM {S}.ccpred9d5953")
print("predictions:", stats)

with open("metrics.json", "w") as f:
    json.dump({"holdout_roc_auc": holdout_auc, "train_roc_auc": train_auc,
               "t_split": t_split, "pred_stats": stats}, f)
print("DONE")
