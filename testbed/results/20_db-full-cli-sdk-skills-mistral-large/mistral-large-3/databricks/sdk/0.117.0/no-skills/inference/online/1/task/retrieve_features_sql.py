#!/usr/bin/env python3
"""
Retrieve feature vectors for the keys in data/lookup_keys.txt via the online store
using SQL queries, and write submission/answers.json.
"""

import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.<run-id>

# Feature table details
FEATURE_TABLE_NAME = "profiles395e7c"
RECORD_KEY = "account_id"

# Paths
lookup_keys_path = "data/lookup_keys.txt"
output_path = "submission/answers.json"

# Initialize WorkspaceClient
w = WorkspaceClient()


def read_lookup_keys():
    """Read account_ids from lookup_keys.txt."""
    with open(lookup_keys_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def retrieve_online_features(account_ids):
    """Retrieve feature vectors for the given account_ids via SQL queries."""
    full_table_name = f"{SCHEMA}.{FEATURE_TABLE_NAME}"
    vectors = {}
    
    # Get a SQL warehouse
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouses available in the workspace.")
    
    for account_id in account_ids:
        try:
            # Query the online store using SQL
            query = f"SELECT f1, f2, f3, f4 FROM {full_table_name} WHERE account_id = '{account_id}'"
            
            result = w.statement_execution.execute_statement(
                warehouse_id=warehouses[0].id,
                catalog=SCHEMA.split(".")[0],
                schema=SCHEMA.split(".")[1],
                statement=query
            )
            
            # Poll for completion
            statement_id = result.statement_id
            while True:
                status = w.statement_execution.get_statement(statement_id)
                if status.status.state == sql.StatementState.SUCCEEDED:
                    break
                elif status.status.state in [sql.StatementState.FAILED, sql.StatementState.CANCELED]:
                    error_msg = getattr(status.status, 'message', 'No error message available')
                    print(f"Query failed for {account_id}: {error_msg}")
                    break
                time.sleep(1)
            
            # Extract results
            if status.status.state == sql.StatementState.SUCCEEDED:
                if hasattr(status, 'result') and hasattr(status.result, 'data_array') and status.result.data_array:
                    feature_values = [
                        float(status.result.data_array[0][0]),  # f1
                        float(status.result.data_array[0][1]),  # f2
                        float(status.result.data_array[0][2]),  # f3
                        float(status.result.data_array[0][3])   # f4
                    ]
                    vectors[account_id] = feature_values
        except Exception as e:
            print(f"Error retrieving features for {account_id}: {e}")
    
    return vectors


def main():
    # Read lookup keys
    account_ids = read_lookup_keys()
    
    # Retrieve feature vectors via SQL
    vectors = retrieve_online_features(account_ids)
    
    # Write the output
    os.makedirs("submission", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"vectors": vectors}, f, indent=2)
    
    print(f"Feature vectors written to {output_path}")


if __name__ == "__main__":
    main()