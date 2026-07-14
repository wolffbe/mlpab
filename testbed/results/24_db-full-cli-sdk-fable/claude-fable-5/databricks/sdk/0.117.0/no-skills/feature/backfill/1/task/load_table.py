import io
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab763a7d
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
TABLE = f"{SCHEMA}.accounts06d84b"
VOLUME = f"{SCHEMA}.staging_accounts06d84b"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/staging_accounts06d84b"
WAREHOUSE_ID = "8a93fc195da2ceb1"

w = WorkspaceClient()


def sql(statement):
    print(">>>", statement.strip().splitlines()[0][:100], "...")
    r = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=statement, wait_timeout="50s"
    )
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {r.status.state} {r.status.error}")
    return r


# 1. Staging volume + upload batches
sql(f"CREATE VOLUME IF NOT EXISTS {VOLUME}")
for i in (1, 2, 3):
    with open(f"data/batch_{i}.csv", "rb") as f:
        w.files.upload(f"{VOLUME_PATH}/batch_{i}.csv", io.BytesIO(f.read()), overwrite=True)
    print("uploaded batch", i)

# 2. Feature table: PK on row_id, updated_at as event-time (TIMESERIES) column
sql(f"DROP TABLE IF EXISTS {TABLE}")
try:
    sql(f"""
    CREATE TABLE {TABLE} (
      row_id STRING NOT NULL,
      status STRING,
      balance DOUBLE,
      updated_at BIGINT NOT NULL,
      CONSTRAINT accounts06d84b_pk PRIMARY KEY (row_id, updated_at TIMESERIES)
    )
    USING DELTA
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
    COMMENT 'Feature table: accounts, record key row_id, event-time updated_at (epoch ms)'
    """)
    print("created table with TIMESERIES PK")
except RuntimeError as e:
    print("TIMESERIES PK failed, falling back to plain PK:", e)
    sql(f"""
    CREATE TABLE {TABLE} (
      row_id STRING NOT NULL,
      status STRING,
      balance DOUBLE,
      updated_at BIGINT NOT NULL,
      CONSTRAINT accounts06d84b_pk PRIMARY KEY (row_id)
    )
    USING DELTA
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
    COMMENT 'Feature table: accounts, record key row_id, event-time updated_at (epoch ms)'
    """)
    print("created table with plain PK")

# 3. Load latest revision per row_id (ties broken by later batch file)
sql(f"""
INSERT INTO {TABLE}
SELECT row_id, status, balance, updated_at FROM (
  SELECT row_id, status, balance, updated_at,
         row_number() OVER (
           PARTITION BY row_id
           ORDER BY updated_at DESC, _metadata.file_name DESC
         ) AS rn
  FROM read_files(
    '{VOLUME_PATH}/',
    format => 'csv',
    header => true,
    schema => 'row_id STRING, status STRING, balance DOUBLE, updated_at BIGINT'
  )
)
WHERE rn = 1
""")

# 4. Verify
r = sql(f"SELECT count(*) AS n, count(DISTINCT row_id) AS d FROM {TABLE}")
print("rows / distinct row_ids:", r.result.data_array)
r = sql(f"SELECT * FROM {TABLE} ORDER BY row_id LIMIT 5")
print("sample:", r.result.data_array)
