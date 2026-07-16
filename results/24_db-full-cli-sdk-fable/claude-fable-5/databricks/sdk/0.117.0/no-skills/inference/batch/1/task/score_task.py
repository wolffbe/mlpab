import databricks.sdk

w = databricks.sdk.WorkspaceClient()
WH = "8a93fc195da2ceb1"
SCHEMA = "workspace.mlpab4fd108"


def run(sql):
    r = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=WH, wait_timeout="50s"
    )
    if str(r.status.state) != "StatementState.SUCCEEDED":
        raise RuntimeError(f"{r.status.state}: {r.status.error}\nSQL: {sql[:300]}")
    return r


lines = open("data/feature_history.csv").read().strip().split("\n")[1:]
print("rows:", len(lines))
w_f1, w_f2, w_f3, bias = -0.8177, 1.3331, 0.0176, -0.2971

run(
    f"CREATE OR REPLACE TABLE {SCHEMA}.feature_history_raw "
    "(account_id STRING, event_time BIGINT, f1 DOUBLE, f2 DOUBLE, f3 DOUBLE)"
)
vals = []
for ln in lines:
    a, t, f1, f2, f3 = ln.split(",")
    vals.append(f"('{a}',{t},{f1},{f2},{f3})")
run(f"INSERT INTO {SCHEMA}.feature_history_raw VALUES " + ",".join(vals))
print("raw table loaded")

run(
    f"""CREATE OR REPLACE TABLE {SCHEMA}.scoresbedd56 (
  account_id STRING NOT NULL,
  score DOUBLE,
  CONSTRAINT scoresbedd56_pk PRIMARY KEY (account_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)"""
)

run(
    f"""INSERT INTO {SCHEMA}.scoresbedd56
SELECT account_id,
       ROUND(1.0 / (1.0 + EXP(-(({w_f1})*f1 + ({w_f2})*f2 + ({w_f3})*f3 + ({bias})))), 6) AS score
FROM (
  SELECT account_id, f1, f2, f3,
         ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) AS rn
  FROM {SCHEMA}.feature_history_raw
  WHERE event_time <= 1773313200000
) WHERE rn = 1"""
)

r = run(f"SELECT COUNT(*), COUNT(DISTINCT account_id) FROM {SCHEMA}.scoresbedd56")
print("scores table count:", r.result.data_array)
r = run(f"SELECT COUNT(DISTINCT account_id) FROM {SCHEMA}.feature_history_raw")
print("distinct accounts in history:", r.result.data_array)
r = run(f"SELECT * FROM {SCHEMA}.scoresbedd56 ORDER BY account_id LIMIT 5")
print(r.result.data_array)
