import databricks.sdk
from databricks.sdk.service.catalog import TableType, DataSourceFormat

w = databricks.sdk.WorkspaceClient()

# First, let's try to use the SQL API to load the data
# We'll create temp tables from the CSV files, then deduplicate and save

# Create a temp table from batch 1
sql1 = """
CREATE OR REPLACE TEMP VIEW temp_batch_1 AS
SELECT * FROM csv.`/Workspace/Users/benedict@hopsworks.ai/mlpabc1ee89/batch_1.csv`
"""

# Create a temp table from batch 2
sql2 = """
CREATE OR REPLACE TEMP VIEW temp_batch_2 AS
SELECT * FROM csv.`/Workspace/Users/benedict@hopsworks.ai/mlpabc1ee89/batch_2.csv`
"""

# Create a temp table from batch 3
sql3 = """
CREATE OR REPLACE TEMP VIEW temp_batch_3 AS
SELECT * FROM csv.`/Workspace/Users/benedict@hopsworks.ai/mlpabc1ee89/batch_3.csv`
"""

# Union and deduplicate
sql4 = """
CREATE OR REPLACE TABLE workspace.mlpabc1ee89.accounts9ad208 AS
WITH all_data AS (
  SELECT * FROM temp_batch_1
  UNION ALL
  SELECT * FROM temp_batch_2
  UNION ALL
  SELECT * FROM temp_batch_3
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY updated_at DESC) as rn
  FROM all_data
)
SELECT row_id, status, balance, updated_at
FROM ranked
WHERE rn = 1
"""

# Execute the SQL statements
warehouse_id = "a832b544eb7dc3fe"  # Serverless Starter Warehouse

for sql in [sql1, sql2, sql3, sql4]:
    result = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        catalog="workspace",
        schema="mlpabc1ee89",
        statement=sql
    )
    print(f"Statement ID: {result.statement_id}, Status: {result.status}")
    
    # Wait and check result
    import time
    time.sleep(2)
    
    try:
        result_data = w.statement_execution.get_statement_result_chunk_n(
            statement_id=result.statement_id,
            chunk_index=0
        )
        print(f"Result: {result_data}")
    except Exception as e:
        print(f"Error getting result: {e}")

print("SQL execution complete")
