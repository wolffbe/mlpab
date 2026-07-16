import os
import csv
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()

schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # e.g. workspace.mlpab53a3ab
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]   # e.g. mlpab53a3ab

catalog_name, schema_name = schema.split(".", 1)
print(f"Using catalog={catalog_name}, schema={schema_name}, prefix={prefix}")

# Read CSV data
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

rows_a = read_csv("data/raw_a.csv")
rows_b = read_csv("data/raw_b.csv")
print(f"raw_a rows: {len(rows_a)}, raw_b rows: {len(rows_b)}")

# Find SQL warehouse to run statements
warehouses = list(w.warehouses.list())
warehouse_id = None
for wh in warehouses:
    if wh.state and wh.state.value in ("RUNNING", "STOPPED"):
        warehouse_id = wh.id
        break
if not warehouse_id and warehouses:
    warehouse_id = warehouses[0].id
print(f"Using warehouse: {warehouse_id}")

def exec_sql(statement):
    """Execute SQL and poll until complete."""
    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        wait_timeout="50s",
    )
    # If still running, poll
    terminal = {StatementState.SUCCEEDED, StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED}
    while resp.status and resp.status.state not in terminal:
        time.sleep(2)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status and resp.status.state != StatementState.SUCCEEDED:
        err = resp.status.error.message if resp.status.error else "unknown error"
        raise RuntimeError(f"SQL error: {err}\nSQL: {statement[:300]}")
    return resp

# --- Step 1: Create raw_a table ---
print("\n=== Creating rawad05474 ===")
exec_sql(f"DROP TABLE IF EXISTS `{catalog_name}`.`{schema_name}`.`rawad05474`")

# Build INSERT values for raw_a
vals_a = ", ".join(f"('{r['row_id']}', {float(r['a_val'])})" for r in rows_a)
exec_sql(f"""
CREATE TABLE `{catalog_name}`.`{schema_name}`.`rawad05474` (
  row_id STRING NOT NULL,
  a_val DOUBLE
)
USING DELTA
TBLPROPERTIES ('delta.minReaderVersion' = '1', 'delta.minWriterVersion' = '2')
""")
exec_sql(f"INSERT INTO `{catalog_name}`.`{schema_name}`.`rawad05474` VALUES {vals_a}")
print("rawad05474 created and loaded.")

# --- Step 2: Create raw_b table ---
print("\n=== Creating rawbd05474 ===")
exec_sql(f"DROP TABLE IF EXISTS `{catalog_name}`.`{schema_name}`.`rawbd05474`")

vals_b = ", ".join(f"('{r['row_id']}', {float(r['b_val'])})" for r in rows_b)
exec_sql(f"""
CREATE TABLE `{catalog_name}`.`{schema_name}`.`rawbd05474` (
  row_id STRING NOT NULL,
  b_val DOUBLE
)
USING DELTA
TBLPROPERTIES ('delta.minReaderVersion' = '1', 'delta.minWriterVersion' = '2')
""")
exec_sql(f"INSERT INTO `{catalog_name}`.`{schema_name}`.`rawbd05474` VALUES {vals_b}")
print("rawbd05474 created and loaded.")

# --- Step 3: Create derived table with lineage ---
print("\n=== Creating derivedd05474 ===")
exec_sql(f"DROP TABLE IF EXISTS `{catalog_name}`.`{schema_name}`.`derivedd05474`")

# Use CTAS with JOIN to establish lineage and compute col_sum
exec_sql(f"""
CREATE TABLE `{catalog_name}`.`{schema_name}`.`derivedd05474`
USING DELTA
AS
SELECT
  a.row_id,
  ROUND(a.a_val + b.b_val, 6) AS col_sum
FROM `{catalog_name}`.`{schema_name}`.`rawad05474` a
INNER JOIN `{catalog_name}`.`{schema_name}`.`rawbd05474` b
  ON a.row_id = b.row_id
ORDER BY a.row_id
""")
print("derivedd05474 created.")

# Verify row count
result = exec_sql(f"SELECT COUNT(*) as cnt FROM `{catalog_name}`.`{schema_name}`.`derivedd05474`")
if result.result and result.result.data_array:
    print(f"derivedd05474 row count: {result.result.data_array[0][0]}")

# --- Step 4: Create online table for low-latency access ---
print("\n=== Creating online table for derivedd05474 ===")
online_table_name = f"{catalog_name}.{schema_name}.derivedd05474_online"

try:
    # Try to delete existing
    w.online_tables.delete(name=online_table_name)
    print("Deleted existing online table.")
except Exception as e:
    print(f"No existing online table to delete: {e}")

try:
    from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
    spec = OnlineTableSpec(
        source_table_full_name=f"{catalog_name}.{schema_name}.derivedd05474",
        primary_key_columns=["row_id"],
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
    )
    online_table = w.online_tables.create(
        name=online_table_name,
        spec=spec,
    )
    print(f"Online table created: {online_table_name}")
    print(f"Online table status: {online_table.status}")
except Exception as e:
    print(f"Online table creation note: {e}")

# --- Step 5: Fetch derived data to write submission CSV ---
print("\n=== Fetching derived data for submission CSV ===")
result = exec_sql(f"SELECT row_id, col_sum FROM `{catalog_name}`.`{schema_name}`.`derivedd05474` ORDER BY row_id")

rows_derived = []
if result.result and result.result.data_array:
    for row in result.result.data_array:
        rows_derived.append({"row_id": row[0], "col_sum": row[1]})

print(f"Fetched {len(rows_derived)} derived rows")

# Write submission CSV
os.makedirs("submission", exist_ok=True)
with open("submission/derivedd05474.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["row_id", "col_sum"])
    writer.writeheader()
    writer.writerows(rows_derived)
print("Written submission/derivedd05474.csv")

# --- Step 6: Write answers.json ---
answers = {
    "derived_from": sorted(["rawad05474", "rawbd05474"])
}
with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)
print("Written submission/answers.json")
print(f"answers: {answers}")

print("\n=== All done ===")
