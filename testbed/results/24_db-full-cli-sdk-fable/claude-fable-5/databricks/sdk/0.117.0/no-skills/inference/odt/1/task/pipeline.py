import csv
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # e.g. workspace.mlpab99356b
WAREHOUSE = "8a93fc195da2ceb1"

w = WorkspaceClient()


def sql(statement, quiet=False):
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE, statement=statement, wait_timeout="50s"
    )
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed ({resp.status.state}): {resp.status.error}\n{statement[:300]}")
    if not quiet:
        print("OK:", statement.strip().split("\n")[0][:100])
    return resp


sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

# --- source tables ---
sql(f"DROP TABLE IF EXISTS {SCHEMA}.requests_src")
sql(f"DROP TABLE IF EXISTS {SCHEMA}.profiles_src")
sql(f"""CREATE TABLE {SCHEMA}.requests_src (
  request_id STRING, account_id STRING, request_lat DOUBLE, request_lon DOUBLE, requested_at STRING)""")
sql(f"""CREATE TABLE {SCHEMA}.profiles_src (
  account_id STRING, home_lat DOUBLE, home_lon DOUBLE, base_score DOUBLE)""")

with open("data/requests.csv") as f:
    reqs = list(csv.DictReader(f))
with open("data/profiles.csv") as f:
    profs = list(csv.DictReader(f))
print(f"loaded {len(reqs)} requests, {len(profs)} profiles")

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for batch in chunks(reqs, 200):
    values = ",".join(
        f"('{r['request_id']}','{r['account_id']}',{r['request_lat']},{r['request_lon']},'{r['requested_at']}')"
        for r in batch
    )
    sql(f"INSERT INTO {SCHEMA}.requests_src VALUES {values}", quiet=True)
print("requests ingested")

values = ",".join(
    f"('{p['account_id']}',{p['home_lat']},{p['home_lon']},{p['base_score']})" for p in profs
)
sql(f"INSERT INTO {SCHEMA}.profiles_src VALUES {values}", quiet=True)
print("profiles ingested")

# --- scored feature table (join + on-demand transform, computed on the platform) ---
sql(f"DROP TABLE IF EXISTS {SCHEMA}.scored288ecf")
sql(f"""CREATE TABLE {SCHEMA}.scored288ecf (
  request_id STRING NOT NULL,
  account_id STRING,
  distance_deg DOUBLE,
  score DOUBLE,
  CONSTRAINT scored288ecf_pk PRIMARY KEY (request_id))
TBLPROPERTIES (delta.enableChangeDataFeed = true)""")

sql(f"""INSERT INTO {SCHEMA}.scored288ecf
SELECT request_id, account_id, distance_deg, round(base_score - 0.1 * distance_deg, 6) AS score
FROM (
  SELECT r.request_id, r.account_id, p.base_score,
         round(sqrt(power(r.request_lat - p.home_lat, 2) + power(r.request_lon - p.home_lon, 2)), 6) AS distance_deg
  FROM {SCHEMA}.requests_src r
  JOIN {SCHEMA}.profiles_src p ON r.account_id = p.account_id
)""")

resp = sql(f"SELECT count(*), count(DISTINCT request_id) FROM {SCHEMA}.scored288ecf")
print("row counts (total, distinct request_id):", resp.result.data_array[0])

resp = sql(f"SELECT * FROM {SCHEMA}.scored288ecf ORDER BY request_id LIMIT 5")
for row in resp.result.data_array:
    print(row)
