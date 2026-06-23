#!/usr/bin/env python3
"""Script to execute the task on Hopsworks platform."""
import os
import json
import hopsworks

# Connect to Hopsworks
hopsworks.login()
fs = hopsworks.fs

# Step 1: Upload the data and script to the platform
# First, let's create a directory for our job
job_dir = "Jobs/trainjob0053a1"
fs.mkdir(job_dir, exist_ok=True)

# Upload the training script
script_local = "data/train_model.py"
script_remote = f"{job_dir}/train_model.py"
fs.upload(script_local, script_remote)

# Upload train.csv and score.csv
fs.upload("data/train.csv", f"{job_dir}/train.csv")
fs.upload("data/score.csv", f"{job_dir}/score.csv")

# Step 2: Create and run the job
import hopsworks

# Get the project
project = hopsworks.project()

# Create a Python job
job_api = project.get_job_api()

# Define the job configuration
job_name = "trainjob0053a1"
job_config = {
    "name": job_name,
    "executable": "train_model.py",
    "input": ["train.csv", "score.csv"],
    "output": ["predictions.csv"],
    "args": [],
    "local_logdir": False,
    "log_level": "INFO",
    "secret_envs": {},
    "env_vars": {},
    "volumes": {},
    "volumes_mounts": {},
    "setup_file": None,
    "conda_env": None,
    "interpreter": "python3",
    "dependencies_dir": None,
    "cpu": 1,
    "memory": 1024,
    "gpus": 0,
    "instance_type": "default",
    "shm": False,
    "command": None,
    "commands": None,
    "schedule": None,
    "action": "train",
    "app_resource_name": None,
    "app_entrypoint": None,
    "worker": None,
    "workers": 1,
    "distributed": "False",
    "engine": "python",
    "engine_instance_type": None,
    "engine_instance_num": None,
    "engine_instance_cpu": None,
    "engine_instance_memory": None,
    "engine_instance_gpus": None,
    "engine_instance_shm": None,
    "engine_instance_volumes": {},
    "engine_instance_volumes_mounts": {},
    "engine_instance_secret_envs": {},
    "engine_instance_env_vars": {},
    "engine_instance_conda_env": None,
    "engine_instance_dependencies_dir": None,
    "engine_instance_command": None,
    "engine_instance_commands": None,
}

# Actually, let's use a simpler approach - use the Job class
from hopsworks.job import Job

# Create the job
job = Job(
    name=job_name,
    executable="train_model.py",
    project_name=project.get_name(),
    input=["train.csv", "score.csv"],
    output=["predictions.csv"],
    args=[],
    local_logdir=False,
    log_level="INFO",
    cpu=1,
    memory=1024,
    gpus=0,
)

# Save the job
job.save()

# Run the job
print(f"Starting job: {job_name}")
job.run(arguments=[], wait=True, sync=True, timeout=3600)
print(f"Job completed: {job_name}")

# Step 3: Load predictions into a feature table
# First, let's find the predictions.csv from the job output
# The job output should be in the job's output directory

# Get the feature store API
fs_api = project.get_feature_store_api()

# Create a feature group
fg_name = "predictions0053a1"
fg_version = 1

# Check if feature group exists, if not create it
try:
    fg = fs_api.get_feature_group(fg_name, version=fg_version)
    print(f"Feature group {fg_name} v{fg_version} already exists")
except:
    # Create the feature group
    fg = fs_api.create_feature_group(
        name=fg_name,
        version=fg_version,
        description="Predictions from trainjob0053a1",
        primary_key=["row_id"],
        online_enabled=True,
    )
    print(f"Created feature group {fg_name} v{fg_version}")

# Now we need to get the predictions.csv from the job output
# The job output should be in Resources/jobs/<job_id>/output/predictions.csv
# Let's list the job's output directory
job_id = job.get_id()
output_path = f"Resources/jobs/{job_id}/output/predictions.csv"

print(f"Looking for predictions at: {output_path}")

# Check if the file exists
if fs.exists(output_path):
    print(f"Found predictions.csv at {output_path}")
    # Download it locally to inspect
    local_preds = "predictions.csv"
    fs.download(output_path, local_preds)
    
    # Read and insert into feature group
    import pandas as pd
    preds_df = pd.read_csv(local_preds)
    
    # Insert into feature group
    fg.insert(preds_df, write_options={"wait_for_job": True})
    print(f"Inserted predictions into feature group {fg_name}")
else:
    print(f"Predictions not found at {output_path}")
    # Try alternative paths
    alternative_paths = [
        f"Jobs/{job_name}/output/predictions.csv",
        f"Jobs/{job_name}/predictions.csv",
        f"Resources/jobs/{job_name}/output/predictions.csv",
    ]
    for path in alternative_paths:
        if fs.exists(path):
            print(f"Found predictions.csv at {path}")
            fs.download(path, "predictions.csv")
            preds_df = pd.read_csv("predictions.csv")
            fg.insert(preds_df, write_options={"wait_for_job": True})
            print(f"Inserted predictions into feature group {fg_name}")
            break
    else:
        print("Could not find predictions.csv")

# Step 4: Enable online access (should already be enabled in creation)
# Make sure online access is enabled
fg = fs_api.get_feature_group(fg_name, version=fg_version)
if not fg.online_enabled:
    fg.enable_online()
    print(f"Enabled online access for {fg_name}")
else:
    print(f"Online access already enabled for {fg_name}")

# Step 5: Write submission/answers.json
answers = {"job_name": job_name}
with open("submission/answers.json", "w") as f:
    json.dump(answers, f)
print(f"Written submission/answers.json: {answers}")

print("Task completed successfully!")
