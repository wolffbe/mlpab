"""
Script to run the provided training script as a job on Hopsworks, 
load its predictions into a feature table, and enable online access.
"""
import hopsworks
import pandas as pd
import os
import json

# Authenticate with Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
jobs_api = project.get_jobs_api()

# Define job name and script path
job_name = "trainjobd2a49d"
script_path = "data/train_model.py"

# List all jobs to check if the job already exists
jobs = jobs_api.get_jobs()
job = next((j for j in jobs if j.name == job_name), None)

if job is not None:
    print(f"Job {job_name} already exists. Deleting it.")
    job.delete()

# Get default Python job configuration
job_config = jobs_api.get_configuration(type="PYTHON")

# Update the job configuration
job_config["appPath"] = script_path
job_config["name"] = job_name
job_config["jobType"] = "PYTHON"  # Explicitly set jobType
job_config["runConfig"] = {
    "filesToAttach": ["data/train.csv", "data/score.csv"]
}

# Create the job using the JobApi's create method with a dictionary wrapper
class DictWrapper:
    def __init__(self, data):
        self.data = data
    
    def json(self):
        import json
        return json.dumps(self.data)

job_conf_wrapper = DictWrapper(job_config)
job = jobs_api.create(name=job_name, job_conf=job_conf_wrapper)
print(f"Job {job_name} created.")

# Launch the job
execution = job.run()
execution = job.run()
print(f"Job {job_name} started with execution ID: {execution.id}")

# Wait for job completion
execution.wait()
print(f"Job {job_name} completed with state: {execution.state}")

# Retrieve predictions output
predictions_path = "predictions.csv"
execution.download_file(predictions_path)
print(f"Downloaded predictions to {predictions_path}")

# Read predictions into a DataFrame
predictions_df = pd.read_csv(predictions_path)
print(f"Predictions DataFrame shape: {predictions_df.shape}")

# Create or update feature table
feature_table_name = "predictionsd2a49d"
feature_table_version = 1

# Create feature group
fg = fs.create_feature_group(
    name=feature_table_name,
    version=feature_table_version,
    description=f"Predictions from job {job_name}",
    primary_key=["row_id"],
    online_enabled=True  # Enable online access
)

# Insert predictions into feature group
fg.insert(predictions_df)
print(f"Feature table {feature_table_name} version {feature_table_version} created and populated.")

# Write submission file
submission_data = {"job_name": job_name}
submission_path = "submission/answers.json"
with open(submission_path, "w") as f:
    json.dump(submission_data, f)

print(f"Submission file written to {submission_path}")