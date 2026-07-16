#!/usr/bin/env python3
"""Test simple job creation."""

import hopsworks
import os
import json
import time

# Connect to Hopsworks
hopsworks.login()
project = hopsworks.get_current_project()

# Upload files
dataset_api = project.get_dataset_api()

# First, clean up - delete old files
# Upload train.csv
dataset_api.upload("data/train.csv", "/Resources/train.csv", overwrite=True)
# Upload score.csv
dataset_api.upload("data/score.csv", "/Resources/score.csv", overwrite=True)
# Upload train_model.py
dataset_api.upload("data/train_model.py", "/Resources/train_model.py", overwrite=True)

# Create job
job_api = project.get_job_api()
config = job_api.get_configuration('PYTHON')
config['appPath'] = "train_model.py"
config['workingDir'] = "/Resources"

job = job_api.create_job("trainjob0053a1", config)
print(f"Job created: {job.name}")

# Run job
execution = job.run()
print(f"Job running: {execution.id}")

# Wait for completion
while True:
    execution = job.get_executions()[0]
    state = execution.state
    print(f"State: {state}")
    if state in ['FINISHED', 'FAILED', 'KILLED']:
        break
    time.sleep(5)

print(f"Final state: {state}")

if state == 'FINISHED':
    # Download predictions
    dataset_api.download("/Resources/predictions.csv", overwrite=True)
    
    # Create feature table
    import pandas as pd
    predictions_df = pd.read_csv("predictions.csv")
    feature_store = project.get_feature_store()
    
    fg = feature_store.create_feature_group(
        name="predictions0053a1",
        version=1,
        primary_key=["row_id"],
        description="Predictions from trainjob0053a1",
        online_enabled=True
    )
    fg.insert(predictions_df, wait=True)
    
    # Write answers.json
    os.makedirs("submission", exist_ok=True)
    with open("submission/answers.json", "w") as f:
        json.dump({"job_name": "trainjob0053a1"}, f)
    
    print("Done!")
else:
    print(f"Job failed with state: {state}")
