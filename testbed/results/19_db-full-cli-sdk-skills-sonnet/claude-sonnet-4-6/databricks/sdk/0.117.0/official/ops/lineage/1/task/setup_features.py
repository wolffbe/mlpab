import os
import csv
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpabda0279
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]   # mlpabda0279

catalog, schema_name = schema.split(".")
print(f"Catalog: {catalog}, Schema: {schema_name}, Prefix: {prefix}")

# Read raw data
raw_a = []
with open("data/raw_a.csv") as f:
    for row in csv.DictReader(f):
        raw_a.append(row)

raw_b = []
with open("data/raw_b.csv") as f:
    for row in csv.DictReader(f):
        row['b_val'] = row['b_val']
        raw_b.append(row)

print(f"raw_a rows: {len(raw_a)}, raw_b rows: {len(raw_b)}")

# Compute derived table (inner join)
a_dict = {r['row_id']: float(r['a_val']) for r in raw_a}
b_dict = {r['row_id']: float(r['b_val']) for r in raw_b}

common_ids = sorted(set(a_dict.keys()) & set(b_dict.keys()))
derived = [
    {"row_id": rid, "col_sum": round(a_dict[rid] + b_dict[rid], 6)}
    for rid in common_ids
]
print(f"Derived rows (inner join): {len(derived)}")

# Write derived to CSV for local submission backup
import os
os.makedirs("submission", exist_ok=True)
with open("submission/derivedd05474.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["row_id", "col_sum"])
    writer.writeheader()
    writer.writerows(derived)
print("Written submission/derivedd05474.csv")

# ---- Now create tables on Databricks via SQL ----
# Use Databricks SQL statement execution API

def run_sql(statement, wait_timeout="50s"):
    import time
    from databricks.sdk.service.sql import StatementState
    resp = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=get_warehouse_id(),
        wait_timeout=wait_timeout,
    )
    # Poll until done if still running
    terminal = {StatementState.SUCCEEDED, StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED}
    while resp.status.state not in terminal:
        time.sleep(2)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise Exception(f"SQL failed ({resp.status.state}): {resp.status.error}")
    return resp

_warehouse_id = None
def get_warehouse_id():
    global _warehouse_id
    if _warehouse_id:
        return _warehouse_id
    warehouses = list(w.warehouses.list())
    for wh in warehouses:
        if wh.state and wh.state.value in ("RUNNING", "STARTING"):
            _warehouse_id = wh.id
            print(f"Using warehouse: {wh.name} ({wh.id})")
            return _warehouse_id
    # Use first available
    if warehouses:
        _warehouse_id = warehouses[0].id
        print(f"Using warehouse: {warehouses[0].name} ({warehouses[0].id})")
        return _warehouse_id
    raise Exception("No warehouse found")

# Get warehouse
wh_id = get_warehouse_id()

# Create raw_a table
print("Creating rawad05474...")
raw_a_values = ", ".join(f"('{r['row_id']}', {float(r['a_val'])})" for r in raw_a)
run_sql(f"""
CREATE OR REPLACE TABLE {catalog}.{schema_name}.rawad05474 (
    row_id STRING NOT NULL,
    a_val DOUBLE
) TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
)
""")
run_sql(f"""
INSERT INTO {catalog}.{schema_name}.rawad05474 VALUES {raw_a_values}
""")
print(f"rawad05474: inserted {len(raw_a)} rows")

# Create raw_b table
print("Creating rawbd05474...")
raw_b_values = ", ".join(f"('{r['row_id']}', {float(r['b_val'])})" for r in raw_b)
run_sql(f"""
CREATE OR REPLACE TABLE {catalog}.{schema_name}.rawbd05474 (
    row_id STRING NOT NULL,
    b_val DOUBLE
) TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
)
""")
run_sql(f"""
INSERT INTO {catalog}.{schema_name}.rawbd05474 VALUES {raw_b_values}
""")
print(f"rawbd05474: inserted {len(raw_b)} rows")

# Create derived table with lineage tracked via SQL (uses lineage tracking built into Unity Catalog)
print("Creating derivedd05474...")
run_sql(f"""
CREATE OR REPLACE TABLE {catalog}.{schema_name}.derivedd05474 (
    row_id STRING NOT NULL,
    col_sum DOUBLE
) TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
)
""")

# Insert by joining — this creates lineage in Unity Catalog
run_sql(f"""
INSERT INTO {catalog}.{schema_name}.derivedd05474
SELECT
    a.row_id,
    ROUND(a.a_val + b.b_val, 6) AS col_sum
FROM {catalog}.{schema_name}.rawad05474 a
INNER JOIN {catalog}.{schema_name}.rawbd05474 b
    ON a.row_id = b.row_id
ORDER BY a.row_id
""")
print("derivedd05474: created with lineage from rawad05474 and rawbd05474")

# Enable online table for derivedd05474
print("Enabling online table for derivedd05474...")
try:
    from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, PrimaryKeyString

    online_table_name = f"{catalog}.{schema_name}.derivedd05474_online"

    ot = w.online_tables.create(
        name=online_table_name,
        spec=OnlineTableSpec(
            source_table_full_name=f"{catalog}.{schema_name}.derivedd05474",
            primary_key_columns=["row_id"],
            run_triggered={}
        )
    )
    print(f"Online table created: {online_table_name}")
    print(f"Online table status: {ot}")
except Exception as e:
    print(f"Online table creation note: {e}")
    # Try alternative approach
    try:
        resp = w.online_tables.create(
            name=f"{catalog}.{schema_name}.derivedd05474_online",
            spec={
                "source_table_full_name": f"{catalog}.{schema_name}.derivedd05474",
                "primary_key_columns": ["row_id"],
                "run_triggered": {}
            }
        )
        print(f"Online table created (dict spec)")
    except Exception as e2:
        print(f"Online table fallback also failed: {e2}")

print("\nAll done!")
print(f"Tables created in {catalog}.{schema_name}:")
print("  - rawad05474")
print("  - rawbd05474")
print("  - derivedd05474")
