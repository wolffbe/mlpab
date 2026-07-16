import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk.service.sql import StatementState

schema_full = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # e.g. workspace.mlpab4fd108
catalog, schema = schema_full.split(".")
T = 1773313200000
W_F1, W_F2, W_F3, BIAS = -0.8177, 1.3331, 0.0176, -0.2971

w = WorkspaceClient()

# pick a warehouse
wh = next(iter(w.warehouses.list()))
for cand in w.warehouses.list():
    if "serverless" in cand.name.lower():
        wh = cand
print("using warehouse:", wh.name, wh.id)


def sql(stmt):
    r = w.statement_execution.execute_statement(
        warehouse_id=wh.id, statement=stmt, wait_timeout="50s"
    )
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {r.status.state} {r.status.error}")
    return r


# 1. volume + upload csv
try:
    w.volumes.create(catalog_name=catalog, schema_name=schema, name="raw",
                     volume_type=VolumeType.MANAGED)
except Exception as e:
    print("volume create:", e)

vol_path = f"/Volumes/{catalog}/{schema}/raw/feature_history.csv"
with open("data/feature_history.csv", "rb") as f:
    w.files.upload(vol_path, f, overwrite=True)
print("uploaded", vol_path)

# 2. create target table with PK + CDF (needed for online table)
sql(f"""
CREATE OR REPLACE TABLE {schema_full}.scoresbedd56 (
  account_id STRING NOT NULL,
  score DOUBLE,
  CONSTRAINT scoresbedd56_pk PRIMARY KEY (account_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# 3. point-in-time scoring, all on the warehouse
sql(f"""
INSERT INTO {schema_full}.scoresbedd56 (account_id, score)
WITH src AS (
  SELECT account_id,
         CAST(event_time AS BIGINT) AS event_time,
         CAST(f1 AS DOUBLE) AS f1,
         CAST(f2 AS DOUBLE) AS f2,
         CAST(f3 AS DOUBLE) AS f3
  FROM read_files('{vol_path}', format => 'csv', header => true)
),
latest AS (
  SELECT account_id, f1, f2, f3,
         ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) AS rn
  FROM src
  WHERE event_time <= {T}
)
SELECT account_id,
       ROUND(1.0 / (1.0 + EXP(-(({W_F1})*f1 + ({W_F2})*f2 + ({W_F3})*f3 + ({BIAS})))), 6) AS score
FROM latest
WHERE rn = 1
""")

r = sql(f"""
SELECT (SELECT COUNT(*) FROM {schema_full}.scoresbedd56) AS rows,
       (SELECT COUNT(DISTINCT account_id)
          FROM read_files('{vol_path}', format => 'csv', header => true)) AS accounts
""")
print("rows in table / distinct accounts in source:", r.result.data_array)

r = sql(f"SELECT * FROM {schema_full}.scoresbedd56 ORDER BY account_id LIMIT 5")
for row in r.result.data_array:
    print(row)
