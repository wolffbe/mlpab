import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql as dbsql
from databricks.sdk.service.catalog import VolumeType

w = WorkspaceClient()

schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab30f9d3
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]  # mlpab30f9d3
me = w.current_user.me()
user = me.user_name

catalog, schema_name = schema.split(".", 1)

print(f"Schema: {schema}, prefix: {prefix}, user: {user}")

# 1. Create a volume to upload CSVs
volume_name = "churn_data"
print(f"Creating volume {schema}.{volume_name}")
try:
    w.volumes.create(
        catalog_name=catalog,
        schema_name=schema_name,
        name=volume_name,
        volume_type=VolumeType.MANAGED,
    )
    print("Volume created")
except Exception as e:
    if "already exists" in str(e).lower():
        print("Volume already exists")
    else:
        raise

volume_path = f"/Volumes/{catalog}/{schema_name}/{volume_name}"

# 2. Upload CSV files to the volume
data_dir = "./data"
csv_files = [
    "transactions.csv",
    "transactions_late.csv",
    "profiles.csv",
    "activity.csv",
    "account_health.csv",
    "labels.csv",
]

for fname in csv_files:
    local_path = os.path.join(data_dir, fname)
    remote_path = f"{volume_path}/{fname}"
    print(f"Uploading {fname}...")
    with open(local_path, "rb") as f:
        w.files.upload(remote_path, f, overwrite=True)
    print(f"  -> {remote_path}")

print("All files uploaded.")

# 3. Get a SQL warehouse to run queries
warehouses = list(w.warehouses.list())
warehouse = warehouses[0]
warehouse_id = warehouse.id
print(f"Using warehouse: {warehouse.name} ({warehouse_id})")


def run_sql(statement):
    """Execute SQL and wait for result."""
    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        wait_timeout="50s",
    )
    state = resp.status.state.value
    if state == "SUCCEEDED":
        return resp
    if state in ("FAILED", "CANCELED", "CLOSED"):
        raise RuntimeError(f"SQL failed: {resp.status.error}")
    # Poll until done
    stmt_id = resp.statement_id
    for _ in range(300):
        time.sleep(3)
        resp = w.statement_execution.get_statement(stmt_id)
        state = resp.status.state.value
        if state == "SUCCEEDED":
            return resp
        if state in ("FAILED", "CANCELED", "CLOSED"):
            raise RuntimeError(f"SQL failed: {resp.status.error}")
    raise TimeoutError("SQL timed out")


# 4. Create Delta tables from CSVs
print("Creating source tables...")

run_sql(f"""
CREATE OR REPLACE TABLE {schema}.transactions
USING DELTA AS
SELECT * FROM read_files(
  '{volume_path}/transactions.csv',
  format => 'csv',
  header => true,
  inferSchema => true
)
UNION ALL
SELECT * FROM read_files(
  '{volume_path}/transactions_late.csv',
  format => 'csv',
  header => true,
  inferSchema => true
)
""")
print("  transactions table created")

run_sql(f"""
CREATE OR REPLACE TABLE {schema}.profiles
USING DELTA AS
SELECT * FROM read_files(
  '{volume_path}/profiles.csv',
  format => 'csv',
  header => true,
  inferSchema => true
)
""")
print("  profiles table created")

run_sql(f"""
CREATE OR REPLACE TABLE {schema}.activity
USING DELTA AS
SELECT * FROM read_files(
  '{volume_path}/activity.csv',
  format => 'csv',
  header => true,
  inferSchema => true
)
""")
print("  activity table created")

run_sql(f"""
CREATE OR REPLACE TABLE {schema}.account_health
USING DELTA AS
SELECT * FROM read_files(
  '{volume_path}/account_health.csv',
  format => 'csv',
  header => true,
  inferSchema => true
)
""")
print("  account_health table created")

run_sql(f"""
CREATE OR REPLACE TABLE {schema}.labels
USING DELTA AS
SELECT * FROM read_files(
  '{volume_path}/labels.csv',
  format => 'csv',
  header => true,
  inferSchema => true
)
""")
print("  labels table created")

# 5. Build point-in-time correct training dataset
# For each (account_id, label_time) get the most recent value at or before label_time
print("Creating training dataset with point-in-time joins...")

training_table = "churntraining807643"

run_sql(f"""
CREATE OR REPLACE TABLE {schema}.{training_table}
USING DELTA AS
WITH
  labels AS (
    SELECT account_id, label_time, churned FROM {schema}.labels
  ),
  -- Most recent transaction at or before label_time
  txn_ranked AS (
    SELECT
      t.account_id,
      t.event_time,
      t.amount,
      t.balance,
      l.label_time,
      ROW_NUMBER() OVER (PARTITION BY t.account_id, l.label_time ORDER BY t.event_time DESC) AS rn
    FROM {schema}.transactions t
    JOIN labels l ON t.account_id = l.account_id AND t.event_time <= l.label_time
  ),
  txn_latest AS (
    SELECT account_id, label_time, amount, balance FROM txn_ranked WHERE rn = 1
  ),
  -- Most recent profile at or before label_time
  prof_ranked AS (
    SELECT
      p.account_id,
      p.event_time,
      p.credit_score,
      p.tier,
      l.label_time,
      ROW_NUMBER() OVER (PARTITION BY p.account_id, l.label_time ORDER BY p.event_time DESC) AS rn
    FROM {schema}.profiles p
    JOIN labels l ON p.account_id = l.account_id AND p.event_time <= l.label_time
  ),
  prof_latest AS (
    SELECT account_id, label_time, credit_score, tier FROM prof_ranked WHERE rn = 1
  ),
  -- Most recent activity at or before label_time
  act_ranked AS (
    SELECT
      a.account_id,
      a.event_time,
      a.sessions_7d,
      l.label_time,
      ROW_NUMBER() OVER (PARTITION BY a.account_id, l.label_time ORDER BY a.event_time DESC) AS rn
    FROM {schema}.activity a
    JOIN labels l ON a.account_id = l.account_id AND a.event_time <= l.label_time
  ),
  act_latest AS (
    SELECT account_id, label_time, sessions_7d FROM act_ranked WHERE rn = 1
  ),
  -- Most recent account_health at or before label_time
  health_ranked AS (
    SELECT
      h.account_id,
      h.event_time,
      h.health_score,
      l.label_time,
      ROW_NUMBER() OVER (PARTITION BY h.account_id, l.label_time ORDER BY h.event_time DESC) AS rn
    FROM {schema}.account_health h
    JOIN labels l ON h.account_id = l.account_id AND h.event_time <= l.label_time
  ),
  health_latest AS (
    SELECT account_id, label_time, health_score FROM health_ranked WHERE rn = 1
  )
SELECT
  l.account_id,
  l.label_time,
  t.amount,
  t.balance,
  p.credit_score,
  p.tier,
  a.sessions_7d,
  h.health_score,
  l.churned
FROM labels l
LEFT JOIN txn_latest t ON l.account_id = t.account_id AND l.label_time = t.label_time
LEFT JOIN prof_latest p ON l.account_id = p.account_id AND l.label_time = p.label_time
LEFT JOIN act_latest a ON l.account_id = a.account_id AND l.label_time = a.label_time
LEFT JOIN health_latest h ON l.account_id = h.account_id AND l.label_time = h.label_time
ORDER BY l.account_id, l.label_time
""")
print(f"  {training_table} table created")

# 6. Verify
print("\nVerifying training dataset...")
resp = run_sql(f"SELECT COUNT(*) AS cnt FROM {schema}.{training_table}")
count = resp.result.data_array[0][0]
print(f"  Row count: {count}")

resp2 = run_sql(f"SELECT * FROM {schema}.{training_table} LIMIT 3")
print("  Sample rows:")
if resp2.result and resp2.result.data_array:
    cols = [c.name for c in resp2.manifest.schema.columns]
    print("  Columns:", cols)
    for row in resp2.result.data_array:
        print("  ", row)

print(f"\nDone! Training dataset '{training_table}' (version 1) created at {schema}.{training_table}")
