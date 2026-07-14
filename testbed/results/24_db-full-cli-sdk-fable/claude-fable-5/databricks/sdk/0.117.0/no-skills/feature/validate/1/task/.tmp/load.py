import io
import json
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab661ec1
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
WAREHOUSE = "8a93fc195da2ceb1"
TABLE = f"{SCHEMA}.events385469"


def sql(stmt, timeout_s=600):
    r = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE, statement=stmt, wait_timeout="50s"
    )
    start = time.time()
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if time.time() - start > timeout_s:
            raise TimeoutError(stmt[:80])
        time.sleep(3)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"{r.status.state}: {r.status.error}\nSTMT: {stmt[:300]}")
    return r


# 1. Volume + upload CSV
sql(f"CREATE VOLUME IF NOT EXISTS {SCHEMA}.raw")
with open("data/events.csv", "rb") as f:
    w.files.upload(
        f"/Volumes/{CATALOG}/{SCHEMA_NAME}/raw/events.csv", f, overwrite=True
    )
print("uploaded csv")

# 2. Staging table with explicit schema (empty amount -> NULL)
sql(f"DROP TABLE IF EXISTS {SCHEMA}.events385469_staging")
sql(
    f"""
CREATE TABLE {SCHEMA}.events385469_staging AS
SELECT * FROM read_files(
  '/Volumes/{CATALOG}/{SCHEMA_NAME}/raw/events.csv',
  format => 'csv',
  header => true,
  schema => 'row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING'
)
"""
)
r = sql(f"SELECT COUNT(*) FROM {SCHEMA}.events385469_staging")
print("staged rows:", r.result.data_array[0][0])

# 3. Target table with PK + CDF (for online serving), load only valid rows
sql(f"DROP TABLE IF EXISTS {TABLE}")
sql(
    f"""
CREATE TABLE {TABLE} (
  row_id STRING NOT NULL,
  account_id STRING,
  event_time BIGINT,
  amount DOUBLE,
  category STRING,
  CONSTRAINT events385469_pk PRIMARY KEY (row_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
"""
)
valid_pred = (
    "amount IS NOT NULL AND amount >= 0 AND amount <= 10000 "
    "AND category IN ('grocery','travel','salary','rent','other')"
)
sql(
    f"""
INSERT INTO {TABLE}
SELECT row_id, account_id, event_time, amount, category
FROM {SCHEMA}.events385469_staging
WHERE {valid_pred}
"""
)
r = sql(f"SELECT COUNT(*) FROM {TABLE}")
print("loaded rows:", r.result.data_array[0][0])

# 4. Rejected row ids (platform-side)
r = sql(
    f"""
SELECT row_id FROM {SCHEMA}.events385469_staging
WHERE NOT ({valid_pred}) OR amount IS NULL
ORDER BY row_id
"""
)
rejected = [row[0] for row in (r.result.data_array or [])]
print("rejected count:", len(rejected))

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"rejected": rejected}, f, indent=1)
print("wrote submission/answers.json")
