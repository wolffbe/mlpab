#!/usr/bin/env python3
"""
Ingest all provided increments into a Hopsworks feature table and set up a recurring job.
"""

import hopsworks
import pandas as pd
import os
from datetime import datetime, timedelta

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Define feature table metadata
table_name = "incremental97a30c"
version = 1
record_key = ["row_id"]
event_time = "event_time"

# Read all increment files
increment_files = [f"data/{f}" for f in sorted(os.listdir("data/")) if f.startswith("increment_")]
dfs = []
for file in increment_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Combine all increments
combined_df = pd.concat(dfs, ignore_index=True)

# Register or get the feature table
fg = None
try:
    fg = fs.get_feature_group(name=table_name, version=version)
    if fg is None:
        raise Exception("Feature group exists but could not be retrieved.")
    print(f"Feature group {table_name} (v{version}) already exists.")
except:
    fg = fs.create_feature_group(
        name=table_name,
        version=version,
        description="Daily increments of events data",
        primary_key=record_key,
        event_time=event_time,
        online_enabled=True,  # Enable online/real-time access
    )
    if fg is None:
        raise RuntimeError("Feature group creation failed.")
    print(f"Feature group {table_name} (v{version}) created.")

# Ingest data
fg.insert(combined_df, write_options={"wait_for_job": True})

# Set up recurring job for future increments
job_name = "incrementaljob97a30c"
job_description = "Daily ingestion of new increments for incremental97a30c"

# Define the job schedule (daily at 00:00 UTC)
schedule = {
    "start_time": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1),
    "cron_expression": "0 0 * * *",  # Daily at 00:00 UTC
}

# Create or update the job
jobs_api = project.get_jobs_api()
try:
    job = jobs_api.get_job(job_name)
    print(f"Job {job_name} already exists.")
except:
    job = jobs_api.create_python_job(
        name=job_name,
        python_script="ingest_increments.py",
        schedule=schedule["cron_expression"],
    )
    print(f"Job {job_name} created.")

print(f"Feature table {table_name} (v{version}) populated and online access enabled.")
print(f"Recurring job {job_name} set up for daily ingestion.")