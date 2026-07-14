#!/usr/bin/env python3
"""
Test reading workspace files.
"""
import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
from databricks.sdk.service.jobs import Task, NotebookTask

wc = WorkspaceClient()
current_user = wc.current_user.me()
USER_HOME = f'/Users/{current_user.user_name}'
PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabf21a49')
WORKSPACE_PATH = f'{USER_HOME}/{PREFIX}'
WAREHOUSE_ID = 'a832b544eb7dc3fe'

def main():
    print("Creating test notebook...")
    
    # Create simple test notebook
    notebook_content = """# Databricks notebook source
print("Testing workspace file access...")

# Try different approaches
WORKSPACE_PATH = "/Users/benedict@hopsworks.ai/mlpabf21a49"

# Approach 1: Direct workspace path
try:
    df = spark.read.csv(WORKSPACE_PATH + "/transactions.csv", header=True, inferSchema=True)
    print("Success with direct workspace path:", df.count(), "rows")
except Exception as e:
    print("Failed with direct workspace path:", e)

# Approach 2: Using file: protocol
try:
    df = spark.read.csv("file:" + WORKSPACE_PATH + "/transactions.csv", header=True, inferSchema=True)
    print("Success with file: protocol:", df.count(), "rows")
except Exception as e:
    print("Failed with file: protocol:", e)

# Approach 3: Using dbutils
try:
    dbutils.fs.ls(WORKSPACE_PATH)
    print("dbutils.fs.ls successful")
except Exception as e:
    print("dbutils.fs.ls failed:", e)

# Approach 4: Copy to DBFS first
try:
    dbutils.fs.cp(WORKSPACE_PATH + "/transactions.csv", "dbfs:/tmp/test_transactions.csv")
    df = spark.read.csv("dbfs:/tmp/test_transactions.csv", header=True, inferSchema=True)
    print("Success with dbfs copy:", df.count(), "rows")
except Exception as e:
    print("Failed with dbfs copy:", e)
"""
    
    notebook_path = f"{WORKSPACE_PATH}/test_workspace_read"
    
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
    
    try:
        wc.workspace.upload(notebook_path, json.dumps(notebook_json).encode('utf-8'), format=ImportFormat.JUPYTER)
    except Exception as e:
        if "already exists" in str(e):
            print("Notebook already exists, updating...")
        else:
            raise
    
    print(f"Test notebook created at: {notebook_path}")
    
    # Create and run job
    print("Creating and running test job...")
    
    job_name = f"{PREFIX}_test_workspace_read"
    
    # Use serverless warehouse
    task = Task(
        task_key="test_workspace_read",
        notebook_task=NotebookTask(
            notebook_path=notebook_path,
            warehouse_id=WAREHOUSE_ID
        )
    )
    
    job = wc.jobs.create(
        name=job_name,
        tasks=[task]
    )
    
    print(f"Test job created: {job.job_id}")
    
    # Run the job
    run = wc.jobs.run_now(job.job_id)
    print(f"Test job run started: {run.run_id}")
    
    print("Test job is running on the platform.")
    print(f"Test notebook: {notebook_path}")
    print(f"Test job ID: {job.job_id}")
    print(f"Test run ID: {run.run_id}")

if __name__ == "__main__":
    main()
