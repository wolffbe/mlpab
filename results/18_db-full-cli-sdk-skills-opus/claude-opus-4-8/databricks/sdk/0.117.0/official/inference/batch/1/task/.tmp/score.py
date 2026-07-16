import databricks.sdk as s
import os

w = s.WorkspaceClient()
catalog, schema = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
wh = "4dfab06c923fe3cc"
table = f"{catalog}.{schema}.scores3380ed"
vol_path = f"/Volumes/{catalog}/{schema}/ingest/feature_history.csv"

T = 1773489600000
w_f1, w_f2, w_f3, bias = -0.4933, -0.7922, 0.23, -0.051

def run(sql):
    r = w.statement_execution.execute_statement(
        warehouse_id=wh, statement=sql, wait_timeout="50s")
    st = r.status.state.value if r.status and r.status.state else None
    if st != "SUCCEEDED":
        # poll if pending
        import time
        sid = r.statement_id
        while st in ("PENDING", "RUNNING"):
            time.sleep(2)
            r = w.statement_execution.get_statement(sid)
            st = r.status.state.value
    if st != "SUCCEEDED":
        raise RuntimeError(f"FAILED: {st} :: {r.status.error}")
    return r

# Drop if exists, then create the feature table with scores valid at T
run(f"DROP TABLE IF EXISTS {table}")

create_sql = f"""
CREATE TABLE {table} AS
WITH raw AS (
  SELECT account_id,
         CAST(event_time AS BIGINT) AS event_time,
         CAST(f1 AS DOUBLE) AS f1,
         CAST(f2 AS DOUBLE) AS f2,
         CAST(f3 AS DOUBLE) AS f3
  FROM read_files('{vol_path}', format => 'csv', header => true)
),
valid AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) AS rn
  FROM raw
  WHERE event_time <= {T}
)
SELECT account_id,
       ROUND(1.0 / (1.0 + EXP(-(({w_f1})*f1 + ({w_f2})*f2 + ({w_f3})*f3 + ({bias})))), 6) AS score
FROM valid
WHERE rn = 1
"""
run(create_sql)
print("table created")

# Make account_id NOT NULL + PRIMARY KEY so it is a Feature (Engineering) table
run(f"ALTER TABLE {table} ALTER COLUMN account_id SET NOT NULL")
run(f"ALTER TABLE {table} ADD CONSTRAINT scores3380ed_pk PRIMARY KEY (account_id)")
print("primary key added")

r = run(f"SELECT COUNT(*) AS n, MIN(score) AS mn, MAX(score) AS mx FROM {table}")
print("stats:", r.result.data_array)
r = run(f"SELECT * FROM {table} ORDER BY account_id LIMIT 5")
print("sample:", r.result.data_array)
