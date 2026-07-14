"""Script to load predictions into a feature table on Databricks."""
import os
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # e.g., workspace.<run-id>
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")
TABLE_NAME = "predictionsac536a"

# Initialize the WorkspaceClient
w = WorkspaceClient()

# Read predictions.csv
predictions = pd.read_csv("./data/predictions.csv")
print(f"Loaded predictions with {len(predictions)} rows")

# Create a feature table
schema_name, table_name = SCHEMA.split(".")
full_table_name = f"{schema_name}.{table_name}"

# Create the schema if it doesn't exist
w.schemas.create(name=schema_name, catalog_name="workspace")

# Create the feature table
w.tables.create(
    name=TABLE_NAME,
    catalog_name="workspace",
    schema_name=schema_name,
    table_type=catalog.TableType.MANAGED,
    columns=[
        catalog.Column(name="row_id", type_name=catalog.ColumnTypeName.STRING, position=0),
        catalog.Column(name="score", type_name=catalog.ColumnTypeName.DOUBLE, position=1),
    ],
    storage_location=f"dbfs:/user/hive/warehouse/{schema_name}.db/{TABLE_NAME}",
)

# Write data to the feature table
for record in predictions.to_dict("records"):
    w.table_data.insert(
        catalog_name="workspace",
        schema_name=schema_name,
        table_name=TABLE_NAME,
        data=[record],
    )

# Enable online access for the feature table
w.online_tables.create(
    catalog_name="workspace",
    schema_name=schema_name,
    table_name=TABLE_NAME,
    primary_key_columns=["row_id"],
)

print(f"Created feature table {full_table_name}.{TABLE_NAME} with online access")