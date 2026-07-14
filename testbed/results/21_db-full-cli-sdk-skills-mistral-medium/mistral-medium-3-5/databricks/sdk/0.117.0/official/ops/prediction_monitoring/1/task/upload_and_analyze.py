#!/usr/bin/env python3
"""
Script to upload prediction log data to Databricks and analyze it for distribution shift.
"""
import os
from databricks.sdk import WorkspaceClient

# Initialize workspace client
wc = WorkspaceClient()

# Environment variables
SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpab379c4e")
PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpab379c4e")

# Path to data file
DATA_FILE = "data/prediction_log.csv"

# Upload CSV to DBFS
print("Uploading prediction log to DBFS...")
dbfs_path = f"dbfs:/tmp/{PREFIX}_prediction_log.csv"
wc.dbfs.upload(DATA_FILE, dbfs_path, overwrite=True)
print(f"Uploaded to {dbfs_path}")

# Create a table from the CSV
# First, let's create a Delta table
catalog_schema = SCHEMA
TABLE_NAME = f"{PREFIX}_prediction_log"
FULL_TABLE_NAME = f"{catalog_schema}.{TABLE_NAME}"

print(f"Creating table {FULL_TABLE_NAME}...")

# Use SQL to create the table
sql_create = f"""
CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME} 
USING CSV
OPTIONS (
  path "{dbfs_path}",
  header "true",
  inferSchema "true"
)
"""

try:
    wc.statement_execution.execute_statement(
        warehouse_id=wc.config.warehouse_id,
        catalog_name="workspace",
        schema_name=SCHEMA.split('.')[1],
        statement=sql_create,
        timeout_seconds=60
    )
    print(f"Table {FULL_TABLE_NAME} created successfully")
except Exception as e:
    print(f"Error creating table: {e}")
    # Try alternative approach - use files API to upload to workspace
    print("Trying alternative approach...")

# Now analyze the data to find the distribution shift
# We'll look for the first date where predictions consistently shift
print("\nAnalyzing data for distribution shift...")

sql_analyze = f"""
WITH daily_stats AS (
  SELECT 
    DATE(ts) as date,
    AVG(prediction) as avg_prediction,
    STDDEV(prediction) as std_prediction,
    MIN(prediction) as min_prediction,
    MAX(prediction) as max_prediction,
    COUNT(*) as count
  FROM {FULL_TABLE_NAME}
  GROUP BY DATE(ts)
  ORDER BY date
),
shift_detection AS (
  SELECT 
    date,
    avg_prediction,
    std_prediction,
    min_prediction,
    max_prediction,
    count,
    LAG(avg_prediction, 1) OVER (ORDER BY date) as prev_avg,
    LAG(std_prediction, 1) OVER (ORDER BY date) as prev_std,
    -- Calculate the difference in means
    avg_prediction - LAG(avg_prediction, 1) OVER (ORDER BY date) as avg_diff,
    -- Calculate the difference in std
    std_prediction - LAG(std_prediction, 1) OVER (ORDER BY date) as std_diff
  FROM daily_stats
)
SELECT 
  date,
  avg_prediction,
  prev_avg,
  avg_diff,
  std_prediction,
  min_prediction,
  max_prediction,
  count
FROM shift_detection
WHERE avg_diff IS NOT NULL
ORDER BY ABS(avg_diff) DESC
LIMIT 10
"""

try:
    result = wc.statement_execution.execute_statement(
        warehouse_id=wc.config.warehouse_id,
        catalog_name="workspace",
        schema_name=SCHEMA.split('.')[1],
        statement=sql_analyze,
        timeout_seconds=60
    )
    
    # Get the results
    result_id = result.result_id
    print(f"Query submitted with result_id: {result_id}")
    
    # Wait and fetch results
    import time
    time.sleep(2)
    
    results = wc.query_history.get_query(result_id)
    print("Query results:")
    print(results)
    
except Exception as e:
    print(f"Error analyzing data: {e}")
    import traceback
    traceback.print_exc()

print("\nAnalysis complete.")
