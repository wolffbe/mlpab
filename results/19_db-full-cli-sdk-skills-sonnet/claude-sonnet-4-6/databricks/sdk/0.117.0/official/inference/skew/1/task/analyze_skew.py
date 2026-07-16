import os
import json
import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Get environment
schema = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.default")
prefix = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpab")

print(f"Schema: {schema}")
print(f"Prefix: {prefix}")

# Read CSV data
with open("data/training_sample.csv") as f:
    training_data = f.read()

with open("data/serving_log.csv") as f:
    serving_data = f.read()

# Upload files to a volume in the schema
# First, check if a volume exists or create one
catalog, schema_name = schema.split(".", 1)
volume_name = f"{prefix}_skew_analysis"

print(f"Using catalog={catalog}, schema={schema_name}")

# Use Statement Execution API to create tables from inline data
# We'll use a VALUES approach or create external tables

# First let's try to create the tables using SQL with inline data
# Parse training data to create SQL VALUES
training_lines = training_data.strip().split("\n")
serving_lines = serving_data.strip().split("\n")

# Get headers
headers = training_lines[0]
print(f"Headers: {headers}")

# Create training VALUES
training_values = []
for line in training_lines[1:]:
    parts = line.split(",")
    entity_id = parts[0]
    vals = parts[1:]
    training_values.append(f"('{entity_id}', {', '.join(vals)})")

# Create serving VALUES
serving_values = []
for line in serving_lines[1:]:
    parts = line.split(",")
    entity_id = parts[0]
    vals = parts[1:]
    serving_values.append(f"('{entity_id}', {', '.join(vals)})")

print(f"Training rows: {len(training_values)}")
print(f"Serving rows: {len(serving_values)}")

# Create training table
create_train_sql = f"""
CREATE OR REPLACE TABLE {schema}.training_sample AS
SELECT * FROM (VALUES
{', '.join(training_values[:100])}
) AS t(entity_id, f1, f2, f3, f4, f5)
"""

# Split into batches if needed (SQL has limits)
def execute_sql(w, sql, timeout=60):
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=None,  # will auto-select
        wait_timeout=f"{timeout}s",
    )
    return resp

# First, find a warehouse
warehouses = list(w.warehouses.list())
print(f"Found {len(warehouses)} warehouses")
if warehouses:
    warehouse_id = warehouses[0].id
    print(f"Using warehouse: {warehouses[0].name} ({warehouse_id})")
else:
    raise Exception("No warehouses available")

def run_sql(sql, description=""):
    print(f"Running SQL: {description or sql[:80]}")
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
    )
    state = resp.status.state.value
    stmt_id = resp.statement_id
    while state not in ["SUCCEEDED", "FAILED", "CANCELED", "CLOSED"]:
        time.sleep(2)
        resp = w.statement_execution.get_statement(stmt_id)
        state = resp.status.state.value
    if state != "SUCCEEDED":
        raise Exception(f"SQL failed ({state}): {resp.status.error}")
    return resp

# Create training table with all data
# Need to batch the inserts
BATCH_SIZE = 100

# Create table first
run_sql(f"""
CREATE OR REPLACE TABLE {schema}.training_sample (
    entity_id STRING, f1 DOUBLE, f2 DOUBLE, f3 DOUBLE, f4 DOUBLE, f5 DOUBLE
)
""", "Create training_sample table")

# Insert training data in batches
for i in range(0, len(training_values), BATCH_SIZE):
    batch = training_values[i:i+BATCH_SIZE]
    insert_sql = f"""
INSERT INTO {schema}.training_sample VALUES
{', '.join(batch)}
"""
    run_sql(insert_sql, f"Insert training batch {i//BATCH_SIZE + 1}")

print("Training data loaded")

# Create serving table
run_sql(f"""
CREATE OR REPLACE TABLE {schema}.serving_log (
    entity_id STRING, f1 DOUBLE, f2 DOUBLE, f3 DOUBLE, f4 DOUBLE, f5 DOUBLE
)
""", "Create serving_log table")

for i in range(0, len(serving_values), BATCH_SIZE):
    batch = serving_values[i:i+BATCH_SIZE]
    insert_sql = f"""
INSERT INTO {schema}.serving_log VALUES
{', '.join(batch)}
"""
    run_sql(insert_sql, f"Insert serving batch {i//BATCH_SIZE + 1}")

print("Serving data loaded")

# Now analyze skew: compare feature statistics for matching entities
analysis_sql = f"""
WITH joined AS (
    SELECT
        t.entity_id,
        t.f1 AS t_f1, s.f1 AS s_f1,
        t.f2 AS t_f2, s.f2 AS s_f2,
        t.f3 AS t_f3, s.f3 AS s_f3,
        t.f4 AS t_f4, s.f4 AS s_f4,
        t.f5 AS t_f5, s.f5 AS s_f5
    FROM {schema}.training_sample t
    JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
),
diffs AS (
    SELECT
        AVG(ABS(t_f1 - s_f1)) AS diff_f1,
        AVG(ABS(t_f2 - s_f2)) AS diff_f2,
        AVG(ABS(t_f3 - s_f3)) AS diff_f3,
        AVG(ABS(t_f4 - s_f4)) AS diff_f4,
        AVG(ABS(t_f5 - s_f5)) AS diff_f5,
        COUNT(*) AS n_matched
    FROM joined
)
SELECT * FROM diffs
"""

resp = run_sql(analysis_sql, "Compare feature diffs for matched entities")
print("\nFeature diff analysis:")
if resp.result and resp.result.data_array:
    cols = [c.name for c in resp.manifest.schema.columns]
    for row in resp.result.data_array:
        print(dict(zip(cols, row)))
else:
    print("No results")

# Also look at correlation / ratio
ratio_sql = f"""
WITH joined AS (
    SELECT
        t.f1 AS t_f1, s.f1 AS s_f1,
        t.f2 AS t_f2, s.f2 AS s_f2,
        t.f3 AS t_f3, s.f3 AS s_f3,
        t.f4 AS t_f4, s.f4 AS s_f4,
        t.f5 AS t_f5, s.f5 AS s_f5
    FROM {schema}.training_sample t
    JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
)
SELECT
    AVG(s_f1 / NULLIF(t_f1, 0)) AS ratio_f1,
    AVG(s_f2 / NULLIF(t_f2, 0)) AS ratio_f2,
    AVG(s_f3 / NULLIF(t_f3, 0)) AS ratio_f3,
    AVG(s_f4 / NULLIF(t_f4, 0)) AS ratio_f4,
    AVG(s_f5 / NULLIF(t_f5, 0)) AS ratio_f5,
    STDDEV(s_f1 / NULLIF(t_f1, 0)) AS std_ratio_f1,
    STDDEV(s_f2 / NULLIF(t_f2, 0)) AS std_ratio_f2,
    STDDEV(s_f3 / NULLIF(t_f3, 0)) AS std_ratio_f3,
    STDDEV(s_f4 / NULLIF(t_f4, 0)) AS std_ratio_f4,
    STDDEV(s_f5 / NULLIF(t_f5, 0)) AS std_ratio_f5
FROM joined
"""

resp2 = run_sql(ratio_sql, "Ratio analysis")
print("\nRatio analysis:")
if resp2.result and resp2.result.data_array:
    cols = [c.name for c in resp2.manifest.schema.columns]
    for row in resp2.result.data_array:
        result = dict(zip(cols, row))
        for k, v in result.items():
            print(f"  {k}: {v}")
else:
    print("No results")

# Also compare distributions
dist_sql = f"""
SELECT
    'training' AS source,
    AVG(f1) AS mean_f1, STDDEV(f1) AS std_f1,
    AVG(f2) AS mean_f2, STDDEV(f2) AS std_f2,
    AVG(f3) AS mean_f3, STDDEV(f3) AS std_f3,
    AVG(f4) AS mean_f4, STDDEV(f4) AS std_f4,
    AVG(f5) AS mean_f5, STDDEV(f5) AS std_f5
FROM {schema}.training_sample
UNION ALL
SELECT
    'serving' AS source,
    AVG(f1) AS mean_f1, STDDEV(f1) AS std_f1,
    AVG(f2) AS mean_f2, STDDEV(f2) AS std_f2,
    AVG(f3) AS mean_f3, STDDEV(f3) AS std_f3,
    AVG(f4) AS mean_f4, STDDEV(f4) AS std_f4,
    AVG(f5) AS mean_f5, STDDEV(f5) AS std_f5
FROM {schema}.serving_log
"""

resp3 = run_sql(dist_sql, "Distribution comparison")
print("\nDistribution comparison:")
if resp3.result and resp3.result.data_array:
    cols = [c.name for c in resp3.manifest.schema.columns]
    for row in resp3.result.data_array:
        result = dict(zip(cols, row))
        print(f"\n  Source: {result['source']}")
        for k, v in result.items():
            if k != 'source':
                print(f"    {k}: {v}")
else:
    print("No results")

print("\nDone!")
