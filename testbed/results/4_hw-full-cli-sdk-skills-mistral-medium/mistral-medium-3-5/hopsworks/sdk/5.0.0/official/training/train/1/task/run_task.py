#!/usr/bin/env python3
"""Script to run the training job on Hopsworks platform and create feature table."""

import hopsworks
import os
import time
import json

# Step 1: Connect to Hopsworks
hopsworks.login()
project = hopsworks.get_current_project()

# Step 2: Upload data files and training script to the platform
print("Uploading files to platform...")
dataset_api = project.get_dataset_api()

# Upload train.csv
dataset_api.upload("data/train.csv", "/Resources/train.csv", overwrite=True)
# Upload score.csv
dataset_api.upload("data/score.csv", "/Resources/score.csv", overwrite=True)
# Upload train_model.py
dataset_api.upload("train_model_dir/train_model.py", "/Resources/train_model.py", overwrite=True)

print("Files uploaded successfully.")

# Step 3: Create and run the job
print("Creating job...")
job_api = project.get_job_api()

# Get Python job configuration
config = job_api.get_configuration('PYTHON')

# Configure the job
config['appPath'] = "/Resources/train_model.py"
config['jobType'] = "PYTHON"

# Create the job
job = job_api.create_job("trainjob0053a1", config)
print(f"Job created: {job.name} (ID: {job.id})")

# Run the job
print("Running job...")
execution = job.run()
print(f"Job execution started: {execution.id}")

# Wait for job to complete
print("Waiting for job to complete...")
while True:
    execution = job.get_executions()[0]  # Get the latest execution
    state = execution.get_state()
    print(f"Current state: {state}")
    if state in ['FINISHED', 'FAILED', 'KILLED']:
        break
    time.sleep(5)

print(f"Job finished with state: {state}")

if state != 'FINISHED':
    print(f"Job failed with state: {state}")
    raise Exception(f"Job execution failed: {state}")

# Step 4: Download predictions.csv from the job execution
print("Downloading predictions.csv...")
# The predictions.csv should be in the Resources directory after job execution
dataset_api.download("/Resources/predictions.csv", overwrite=True)
print("Predictions downloaded.")

# Step 5: Create feature table from predictions.csv
print("Creating feature table...")
import pandas as pd

# Read predictions
predictions_df = pd.read_csv("predictions.csv")
print(f"Predictions shape: {predictions_df.shape}")
print(f"Predictions columns: {predictions_df.columns.tolist()}")

# Get feature store
feature_store = project.get_feature_store()

# Create or get feature group
fg_name = "predictions0053a1"
fg_version = 1

try:
    fg = feature_store.get_feature_group(fg_name, version=fg_version)
    print(f"Feature group {fg_name} v{fg_version} already exists")
except:
    # Create new feature group with online enabled
    fg = feature_store.create_feature_group(
        name=fg_name,
        version=fg_version,
        primary_key=["row_id"],
        description="Predictions from trainjob0053a1",
        online_enabled=True
    )
    print(f"Created feature group {fg_name} v{fg_version}")

# Insert data into feature group
fg.insert(predictions_df, wait=True)
print("Data inserted into feature group.")

# Step 7: Write answers.json
print("Writing answers.json...")
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"job_name": "trainjob0053a1"}, f)
print("answers.json written.")

print("Task completed successfully!")
