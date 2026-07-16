#!/usr/bin/env python3
"""
Create a simple notebook to test data loading.
"""
import os
import json
from databricks.sdk import WorkspaceClient

# Configuration
SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabf21a49')
PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabf21a49')

wc = WorkspaceClient()
current_user = wc.current_user.me()
USER_HOME = f'/Users/{current_user.user_name}'
WORKSPACE_PATH = f'{USER_HOME}/{PREFIX}'

def main():
    print("Creating simple test notebook...")
    
    # Create a simple notebook that reads CSV from workspace
    notebook_content = f'''# Databricks notebook source
# MAGIC %python

# Read CSV from workspace
print("Reading CSV from workspace...")

# Try different paths
paths_to_try = [
    "/Workspace{USER_HOME}/transactions.csv",
    f"{USER_HOME}/transactions.csv",
    "/dbfs{USER_HOME}/transactions.csv",
    f"file:{USER_HOME}/transactions.csv"
]

for path in paths_to_try:
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        print(f"Success with path: {{path}}")
        print(f"Shape: {{df.count()}} rows, {{len(df.columns)}} columns")
        break
    except Exception as e:
        print(f"Failed with path {{path}}: {{e}}")

# If none worked, try to list workspace files
import os
workspace_files = [f for f in os.listdir(f"{USER_HOME}") if f.endswith('.csv') or f.endswith('.zip')]
print(f"Workspace files: {{workspace_files}}")

# Try to extract from zip if available
if 'transactions.zip' in workspace_files:
    print("Found transactions.zip, trying to extract...")
    import zipfile
    with zipfile.ZipFile(f"{USER_HOME}/transactions.zip", 'r') as zip_ref:
        zip_ref.extractall(f"{USER_HOME}/")
    print("Extracted successfully")
    
    # Now try to read the extracted CSV
    try:
        df = spark.read.csv(f"{USER_HOME}/transactions.csv", header=True, inferSchema=True)
        print(f"Success reading extracted CSV: {{df.count()}} rows")
    except Exception as e:
        print(f"Failed reading extracted CSV: {{e}}")
'''
    
    notebook_path = f"{WORKSPACE_PATH}/test_data_loading"
    
    # Create notebook using workspace API
    notebook_json = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": notebook_content
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    wc.workspace.upload(notebook_path, json.dumps(notebook_json).encode('utf-8'), format='JUPYTER')
    print(f"Notebook created at: {notebook_path}")
    
    # Run the notebook as a job
    print("Running notebook as job...")
    
    job = wc.jobs.create(
        name=f"{PREFIX}_test_data_loading",
        tasks=[
            {
                "task_key": "test_data_loading",
                "notebook_task": {
                    "notebook_path": notebook_path
                },
                "warehouse_id": "a832b544eb7dc3fe"
            }
        ]
    )
    
    print(f"Job created: {job.job_id}")
    
    # Run the job
    run = wc.jobs.run_now(job.job_id)
    print(f"Job run started: {run.run_id}")
    
    # Wait for completion
    import time
    print("Waiting for job to complete...")
    while True:
        run_info = wc.jobs.get_run(run.run_id)
        state = run_info.state.life_cycle_state
        result_state = run_info.state.result_state
        
        if state in ["TERMINATED", "SKIPPED"]:
            print(f"Job completed with state: {state}, result: {result_state}")
            break
        elif state == "INTERNAL_ERROR":
            print(f"Job failed with internal error: {result_state}")
            break
        
        time.sleep(10)
    
    # Get job output
    try:
        output = wc.jobs.get_run_output(run.run_id)
        print(f"Job output: {output}")
    except Exception as e:
        print(f"Could not get job output: {e}")

if __name__ == "__main__":
    main()
