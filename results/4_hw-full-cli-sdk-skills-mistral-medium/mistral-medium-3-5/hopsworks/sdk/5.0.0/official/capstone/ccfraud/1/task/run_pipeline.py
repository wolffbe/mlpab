#!/usr/bin/env python3
"""
Main script to orchestrate the pipeline.
Uploads data and script, then submits a job to run on the platform.
"""

import hopsworks
import os
import time

# Disable SSL verification
os.environ['HOPSWORKS_VERIFY_SSL'] = 'false'

# Connect to Hopsworks
hopsworks.login()
project = hopsworks.get_current_project()
ds_api = project.get_dataset_api()
job_api = project.get_job_api()

print("Connected to Hopsworks project:", project.name)

# ============================================================================
# Step 1: Upload data files and training script to Resources
# ============================================================================
print("\n=== Step 1: Uploading files to Resources ===")

# Upload transactions.csv
print("Uploading transactions.csv...")
ds_api.upload(
    local_path='data/transactions.csv',
    upload_path='/Projects/' + project.name + '/Resources/transactions.csv',
    overwrite=True
)

# Upload score_transactions.csv
print("Uploading score_transactions.csv...")
ds_api.upload(
    local_path='data/score_transactions.csv',
    upload_path='/Projects/' + project.name + '/Resources/score_transactions.csv',
    overwrite=True
)

# Upload training script
print("Uploading train_and_score.py...")
ds_api.upload(
    local_path='train_and_score.py',
    upload_path='/Projects/' + project.name + '/Resources/train_and_score.py',
    overwrite=True
)

print("Files uploaded successfully")

# ============================================================================
# Step 2: Create and run job
# ============================================================================
print("\n=== Step 2: Creating and running job ===")

# Get Python configuration (not Spark)
python_config = job_api.get_configuration("PYTHON")

# Configure the job
job_config = {
    **python_config,
    'appPath': '/Projects/' + project.name + '/Resources/train_and_score.py',
    'arguments': [],
}

# Create the job
job_name = "ccfraud_pipeline_job"
try:
    job = job_api.get_job(job_name)
    print(f"Job {job_name} already exists, deleting and recreating...")
    job.delete()
except:
    pass

job = job_api.create_job(job_name, job_config)
print(f"Created job: {job_name}")

# Run the job
print(f"Starting job {job_name}...")
execution = job.run()
print(f"Job execution ID: {execution.id}")

# Wait for job to complete
print("Waiting for job to complete...")
while True:
    state = execution.get_state()
    print(f"Job state: {state}")
    if state in ['FINISHED', 'FAILED', 'KILLED']:
        break
    time.sleep(10)

if state == 'FINISHED':
    print("Job completed successfully!")
else:
    print(f"Job failed with state: {state}")
    # Try to get logs
    try:
        logs = execution.get_log()
        print("Job logs:")
        print(logs)
    except Exception as e:
        print(f"Could not get logs: {e}")

print("\n=== Pipeline orchestration complete ===")
