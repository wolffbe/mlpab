"""Script to load predictions into a Delta table on Databricks."""
import os
import pandas as pd
from databricks.sdk import WorkspaceClient

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # e.g., workspace.<run-id>
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")
TABLE_NAME = "predictionsac536a"

# Initialize the WorkspaceClient
w = WorkspaceClient()

# Read predictions.csv
predictions = pd.read_csv("./data/predictions.csv")
print(f"Loaded predictions with {len(predictions)} rows")

# Save predictions to a temporary CSV file
predictions.to_csv("./predictions_temp.csv", index=False)

# Upload the CSV to DBFS
w.dbfs.upload("./predictions_temp.csv", f"dbfs:/tmp/{TABLE_NAME}.csv", overwrite=True)
print("Uploaded predictions to DBFS")

# Create a Delta table from the CSV using SQL
query = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
    row_id STRING,
    score DOUBLE
)
USING DELTA
LOCATION 'dbfs:/user/hive/warehouse/{SCHEMA.split('.')[1]}.db/{TABLE_NAME}'
AS SELECT * FROM csv.`dbfs:/tmp/{TABLE_NAME}.csv`
"""

w.statement_execution.execute_statement(
    warehouse_id="<warehouse_id>",  # Replace with a valid warehouse ID or use the default
    catalog="workspace",
    schema=SCHEMA.split(".")[1],
    statement=query,
)

# Enable online access for the table
w.online_tables.create(
    catalog_name="workspace",
    schema_name=SCHEMA.split(".")[1],
    table_name=TABLE_NAME,
    primary_key_columns=["row_id"],
)

print(f"Created Delta table {SCHEMA}.{TABLE_NAME} with online access")