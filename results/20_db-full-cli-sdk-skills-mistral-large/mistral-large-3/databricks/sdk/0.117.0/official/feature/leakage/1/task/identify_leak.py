#!/usr/bin/env python3
"""
Identify the feature that leaks the outcome in the dataset.
"""

import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

# Environment variables
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
table_name = f"{prefix}_leak_detection"
user = os.environ.get("USER", "unknown")

# Initialize WorkspaceClient
w = WorkspaceClient()

# Create a table in Unity Catalog and load the dataset
def create_table_and_load_data():
    # Upload the dataset to a workspace file
    workspace_path = f"/Users/{user}/{prefix}/training_data.csv"
    with open("data/training_data.csv", "rb") as f:
        w.workspace.upload(workspace_path, f.read(), overwrite=True)

    # Create a table from the uploaded data
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {schema}.{table_name}
    USING CSV
    OPTIONS (path "{workspace_path}", header "true", inferSchema "true")
    """
    w.statement_execution.execute_statement(
        warehouse_id=w.warehouses.list()[0].id,
        catalog=schema.split(".")[0],
        schema=schema.split(".")[1],
        statement=create_table_query
    ).result()

    return f"{schema}.{table_name}"

# Compute correlation between each feature and the label
def compute_correlations(table_full_name):
    query = f"""
    SELECT
        corr(f1, label) AS f1_corr,
        corr(f2, label) AS f2_corr,
        corr(f3, label) AS f3_corr,
        corr(f4, label) AS f4_corr,
        corr(f5, label) AS f5_corr,
        corr(f6, label) AS f6_corr
    FROM {table_full_name}
    """
    result = w.statement_execution.execute_statement(
        warehouse_id=w.warehouses.list()[0].id,
        catalog=schema.split(".")[0],
        schema=schema.split(".")[1],
        statement=query
    ).result()

    # Extract correlation values
    correlations = result.as_dict()["result"].get("data_array", [])[0]
    feature_names = ["f1", "f2", "f3", "f4", "f5", "f6"]
    correlation_values = dict(zip(feature_names, correlations))
    
    return correlation_values

# Identify the leaking feature
def identify_leaking_feature(correlations):
    # The leaking feature will have a correlation close to 1 or -1
    leaking_feature = max(correlations.items(), key=lambda x: abs(x[1]))
    return leaking_feature[0]

# Main execution
if __name__ == "__main__":
    table_full_name = create_table_and_load_data()
    correlations = compute_correlations(table_full_name)
    leaking_feature = identify_leaking_feature(correlations)
    
    # Write the result to submission/answers.json
    result = {
        "feature": leaking_feature,
        "evidence": f"Correlation values: {correlations}"
    }
    
    with open("submission/answers.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Leaking feature: {leaking_feature}")