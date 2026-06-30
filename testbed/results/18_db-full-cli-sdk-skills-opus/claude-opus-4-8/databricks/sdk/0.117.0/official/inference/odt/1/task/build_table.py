import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WID = "4dfab06c923fe3cc"
CATALOG, SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
FQ = f"{CATALOG}.{SCHEMA}"
vbase = f"/Volumes/{CATALOG}/{SCHEMA}/staging"


def sql(stmt):
    r = w.statement_execution.execute_statement(stmt, WID, catalog=CATALOG, schema=SCHEMA, wait_timeout="50s")
    sid = r.statement_id
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(sid)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL FAILED: {r.status.state} :: {r.status.error}\n{stmt[:300]}")
    return r


# Staging tables loaded from the uploaded CSVs (platform-side ingestion)
sql(f"""CREATE OR REPLACE TABLE {FQ}.stg_requests AS
SELECT * FROM read_files('{vbase}/requests.csv', format => 'csv', header => true,
  schema => 'request_id STRING, account_id STRING, request_lat DOUBLE, request_lon DOUBLE, requested_at STRING')""")
sql(f"""CREATE OR REPLACE TABLE {FQ}.stg_profiles AS
SELECT * FROM read_files('{vbase}/profiles.csv', format => 'csv', header => true,
  schema => 'account_id STRING, home_lat DOUBLE, home_lon DOUBLE, base_score DOUBLE')""")
print("staging loaded")

# Feature table: PK + Change Data Feed (required for online sync)
sql(f"DROP TABLE IF EXISTS {FQ}.scoredbfc4ef")
sql(f"""CREATE TABLE {FQ}.scoredbfc4ef (
  request_id STRING NOT NULL,
  account_id STRING,
  distance_deg DOUBLE,
  score DOUBLE,
  CONSTRAINT scoredbfc4ef_pk PRIMARY KEY(request_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)""")
print("feature table created")

# On-demand transformation: distance from request coords to stored home, then score.
# Uses the ROUNDED distance when computing score, both rounded to 6 decimals.
sql(f"""INSERT INTO {FQ}.scoredbfc4ef
WITH joined AS (
  SELECT r.request_id, r.account_id,
         ROUND(SQRT(POWER(r.request_lat - p.home_lat, 2) + POWER(r.request_lon - p.home_lon, 2)), 6) AS distance_deg,
         p.base_score
  FROM {FQ}.stg_requests r
  JOIN {FQ}.stg_profiles p ON r.account_id = p.account_id
)
SELECT request_id, account_id, distance_deg,
       ROUND(base_score - 0.1 * distance_deg, 6) AS score
FROM joined""")
print("feature table populated")

# Verify
r = sql(f"SELECT COUNT(*) AS n FROM {FQ}.scoredbfc4ef")
print("row count:", r.result.data_array[0][0])
r = sql(f"SELECT * FROM {FQ}.scoredbfc4ef ORDER BY request_id LIMIT 3")
print("sample:", r.result.data_array)
print("DONE build")
