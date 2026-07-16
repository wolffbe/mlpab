#!/usr/bin/env python3
"""
Identify the feature that leaks the outcome in the training data.
Uses Databricks SDK to analyze feature correlations with the label.
"""

import os
import json
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, sql

# Get environment variables
schema_name = os.environ["MLPAB_DATABRICKS_SCHEMA"]
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]

# Initialize Databricks client
w = WorkspaceClient()

# Read the data
print("Reading data...")
df = pd.read_csv("data/training_data.csv")
features = [col for col in df.columns if col not in ["row_id", "label"]]

# Create a temporary volume for data upload
volume_name = f"{prefix}_leak_analysis_volume"
try:
    w.volumes.create(
        catalog_name="workspace",
        schema_name=schema_name,
        name=volume_name,
        volume_type=catalog.VolumeType.MANAGED
    )
except Exception as e:
    print(f"Volume may already exist: {e}")

# Upload the data file to the volume
volume_path = f"/Volumes/workspace/{schema_name}/{volume_name}/training_data.csv"
with open("data/training_data.csv", "rb") as f:
    w.files.upload(volume_path, f, overwrite=True)

# Create a table from the uploaded data
table_name = f"{prefix}_leak_analysis_table"
try:
    w.tables.create(
        catalog_name="workspace",
        schema_name=schema_name,
        name=table_name,
        table_type=catalog.TableType.MANAGED,
        data_source_format=catalog.DataSourceFormat.CSV,
        columns=[
            catalog.Column(name=col, type_name=catalog.ColumnTypeName.DOUBLE)
            if col != "row_id" else 
            catalog.Column(name=col, type_name=catalog.ColumnTypeName.STRING)
            for col in df.columns
        ],
        storage_location=f"/Volumes/workspace/{schema_name}/{volume_name}"
    )
except Exception as e:
    print(f"Table may already exist: {e}")

# Wait for table to be ready
import time
time.sleep(10)  # Give it time to process

# Analyze feature correlations with the label
print("Analyzing feature correlations...")
query = f"""
SELECT 
    corr(f1, label) as f1_corr,
    corr(f2, label) as f2_corr,
    corr(f3, label) as f3_corr,
    corr(f4, label) as f4_corr,
    corr(f5, label) as f5_corr,
    corr(f6, label) as f6_corr,
    abs(corr(f1, label)) as f1_abs_corr,
    abs(corr(f2, label)) as f2_abs_corr,
    abs(corr(f3, label)) as f3_abs_corr,
    abs(corr(f4, label)) as f4_abs_corr,
    abs(corr(f5, label)) as f5_abs_corr,
    abs(corr(f6, label)) as f6_abs_corr
FROM workspace.{schema_name}.{table_name}
WHERE row_id != 'row_id'
"""

try:
    results = w.statement_execution.execute_statement(
        warehouse_id=w.warehouses.list()[0].id,
        catalog="workspace",
        schema=schema_name,
        statement=query
    ).result()
    
    # Get the correlation results
    corr_data = results.to_pandas()
    
    # Find the feature with the highest absolute correlation
    corr_values = {
        'f1': corr_data['f1_abs_corr'].iloc[0],
        'f2': corr_data['f2_abs_corr'].iloc[0],
        'f3': corr_data['f3_abs_corr'].iloc[0],
        'f4': corr_data['f4_abs_corr'].iloc[0],
        'f5': corr_data['f5_abs_corr'].iloc[0],
        'f6': corr_data['f6_abs_corr'].iloc[0]
    }
    
    leaking_feature = max(corr_values.items(), key=lambda x: x[1])[0]
    
    # Prepare the answer
    answer = {
        "feature": leaking_feature,
        "evidence": f"Feature {leaking_feature} has the highest absolute correlation ({corr_values[leaking_feature]:.4f}) with the label, suggesting it leaks outcome information."
    }
    
    # Write the answer to submission/answers.json
    with open("submission/answers.json", "w") as f:
        json.dump(answer, f, indent=2)
    
    print(f"Identified leaking feature: {leaking_feature}")
    print(f"Answer written to submission/answers.json")
    
except Exception as e:
    print(f"Error during analysis: {e}")
    # Fallback to local analysis if Databricks fails
    if "corr_data" not in locals():
        print("Falling back to local correlation analysis...")
        corr_matrix = df.corr()
        corr_with_label = corr_matrix["label"].abs().sort_values(ascending=False)
        leaking_feature = corr_with_label.index[1]  # Skip label itself
        
        answer = {
            "feature": leaking_feature,
            "evidence": f"Feature {leaking_feature} has the highest absolute correlation ({corr_with_label[leaking_feature]:.4f}) with the label, suggesting it leaks outcome information."
        }
        
        with open("submission/answers.json", "w") as f:
            json.dump(answer, f, indent=2)
        
        print(f"Identified leaking feature (local fallback): {leaking_feature}")
        print(f"Answer written to submission/answers.json")