import io
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk.service.sql import StatementState

WAREHOUSE_ID = "a832b544eb7dc3fe"
FULL_SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpabac8a28
CATALOG, SCHEMA = FULL_SCHEMA.split(".")
TABLE = f"{FULL_SCHEMA}.customers03eedc"
VOLUME = "staging03eedc"
VOL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

w = WorkspaceClient()


def sql(statement, quiet=False):
    resp = w.statement_execution.execute_statement(
        statement=statement, warehouse_id=WAREHOUSE_ID, wait_timeout="50s"
    )
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed [{resp.status.state}]: {resp.status.error}\n{statement}")
    rows = resp.result.data_array if resp.result and resp.result.data_array else []
    if not quiet:
        print(f"OK: {statement.splitlines()[0][:80]} -> {rows[:3] if rows else ''}")
    return rows


# --- stage the CSV exports in a UC volume ---
try:
    w.volumes.create(catalog_name=CATALOG, schema_name=SCHEMA, name=VOLUME,
                     volume_type=VolumeType.MANAGED)
    print("volume created")
except Exception as e:
    print(f"volume: {e}")

for local, remote in [("data/initial_export.csv", "initial_export.csv"),
                      ("data/reload/new_export.csv", "new_export.csv")]:
    with open(local, "rb") as f:
        w.files.upload(f"{VOL_PATH}/{remote}", io.BytesIO(f.read()), overwrite=True)
    print(f"uploaded {remote}")

# --- version 1: feature table on the original schema, load initial export ---
sql(f"DROP TABLE IF EXISTS {TABLE}")
sql(f"""CREATE TABLE {TABLE} (
  row_id STRING NOT NULL,
  name STRING,
  balance_eur DOUBLE,
  updated_at BIGINT NOT NULL,
  CONSTRAINT customers03eedc_pk PRIMARY KEY (row_id)
) COMMENT 'Feature table customers03eedc version 1; record key row_id, event time updated_at (epoch ms)'
TBLPROPERTIES (delta.enableChangeDataFeed = true)""")
sql(f"""INSERT INTO {TABLE}
SELECT row_id, name, balance_eur, updated_at
FROM read_files('{VOL_PATH}/initial_export.csv', format => 'csv', header => true,
  schema => 'row_id STRING, name STRING, balance_eur DOUBLE, updated_at BIGINT')""")
print("v1 count:", sql(f"SELECT count(*) FROM {TABLE}", quiet=True))

# --- version 2: full reload — re-create the table from scratch on the new schema ---
sql(f"DROP TABLE {TABLE}")
sql(f"""CREATE TABLE {TABLE} (
  row_id STRING NOT NULL,
  full_name STRING,
  balance DOUBLE,
  currency STRING,
  updated_at BIGINT NOT NULL,
  CONSTRAINT customers03eedc_pk PRIMARY KEY (row_id)
) COMMENT 'Feature table customers03eedc version 2 (full reload after breaking schema change); record key row_id, event time updated_at (epoch ms)'
TBLPROPERTIES (delta.enableChangeDataFeed = true)""")
sql(f"""INSERT INTO {TABLE}
SELECT row_id, full_name, balance, currency, updated_at
FROM read_files('{VOL_PATH}/new_export.csv', format => 'csv', header => true,
  schema => 'row_id STRING, full_name STRING, balance DOUBLE, currency STRING, updated_at BIGINT')""")

print("v2 count:", sql(f"SELECT count(*) FROM {TABLE}", quiet=True))
print("v2 columns:", sql(f"SELECT column_name, data_type FROM {CATALOG}.information_schema.columns "
                         f"WHERE table_schema = '{SCHEMA}' AND table_name = 'customers03eedc' "
                         f"ORDER BY ordinal_position", quiet=True))
print("v2 sample:", sql(f"SELECT * FROM {TABLE} ORDER BY row_id LIMIT 3", quiet=True))
