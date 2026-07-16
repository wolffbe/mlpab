import os
import json
import csv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *
from databricks.sdk.service.sql import *

# Initialize WorkspaceClient
w = WorkspaceClient()

# Environment variables
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")
catalog_name, schema = schema_name.split(".")
table_name = "profiles395e7c"
feature_table_full_name = f"{catalog_name}.{schema}.{table_name}"

# Step 1: Create a feature table in the specified schema
print("Creating feature table...")
try:
    # Get warehouse ID
    warehouse_id = list(w.warehouses.list())[0].id
    
    # Create the table using SQL
    w.statement_execution.execute_statement(
        catalog=catalog_name,
        schema=schema,
        statement=f"""
        CREATE TABLE {feature_table_full_name} (
            account_id STRING,
            f1 FLOAT,
            f2 FLOAT,
            f3 FLOAT,
            f4 FLOAT
        )
        USING DELTA
        """,
        warehouse_id=warehouse_id,
    )
    print(f"Table created: {feature_table_full_name}")
except Exception as e:
    print(f"Error creating table: {e}")
    raise

# Step 2: Load data into the table
print("Loading data into table...")
try:
    # Read the CSV file
    with open("data/features.csv", "r") as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    
    # Write data to the table using SQL
    for row in data:
        w.statement_execution.execute_statement(
            catalog=catalog_name,
            schema=schema,
            statement=f"INSERT INTO {feature_table_full_name} VALUES ('{row['account_id']}', {row['f1']}, {row['f2']}, {row['f3']}, {row['f4']})",
            warehouse_id=warehouse_id,
        )
    print("Data loaded into table.")
except Exception as e:
    print(f"Error loading data into table: {e}")
    raise

# Step 3: Retrieve feature vectors for lookup keys using SQL
print("Retrieving feature vectors...")
try:
    # Read lookup keys
    with open("data/lookup_keys.txt", "r") as f:
        lookup_keys = [line.strip() for line in f if line.strip()]
    
    # Retrieve feature vectors using SQL
    vectors = {}
    for key in lookup_keys:
        try:
            result = w.statement_execution.execute_statement(
                catalog=catalog_name,
                schema=schema,
                statement=f"SELECT f1, f2, f3, f4 FROM {feature_table_full_name} WHERE account_id = '{key}'",
                warehouse_id=warehouse_id,
            )
            
            # Extract the result
            if result.result.data_array:
                row = result.result.data_array[0]
                vectors[key] = [row[0], row[1], row[2], row[3]]
        except Exception as e:
            print(f"Error retrieving features for key {key}: {e}")
    
    # Write results to submission/answers.json
    with open("submission/answers.json", "w") as f:
        json.dump({"vectors": vectors}, f, indent=2)
    print("Feature vectors retrieved and written to submission/answers.json.")
    
except Exception as e:
    print(f"Error retrieving feature vectors: {e}")
    raise