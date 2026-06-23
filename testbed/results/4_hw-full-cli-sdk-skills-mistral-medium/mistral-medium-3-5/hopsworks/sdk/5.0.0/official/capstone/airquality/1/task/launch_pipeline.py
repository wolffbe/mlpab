#!/usr/bin/env python3
"""
Launcher script to upload data and submit the full pipeline job to Hopsworks.
"""

import hopsworks

print("Launching Air Quality FTI Pipeline on Hopsworks")
print("=" * 60)

# Connect to Hopsworks
project = hopsworks.login()
ds_api = project.get_dataset_api()
job_api = project.get_job_api()

# Upload data files and script
print("\n1. Uploading files to Hopsworks...")
ds_api.upload('data/airquality_history.csv', 'Resources/airquality_history.csv', overwrite=True)
ds_api.upload('data/forecast_days.csv', 'Resources/forecast_days.csv', overwrite=True)
ds_api.upload('full_pipeline_job.py', 'Resources/full_pipeline_job.py', overwrite=True)
print("   Files uploaded")

# Create job configuration
print("\n2. Creating job...")
config = job_api.get_configuration('PYTHON')
config['appPath'] = 'Resources/full_pipeline_job.py'
config['resourceConfig']['cores'] = 4
config['resourceConfig']['memory'] = 16384

# Create job
job = job_api.create_job(
    name='airq_full_pipeline_2ce555',
    config=config
)
print("   Job created")

# Run job
print("\n3. Launching job...")
job.run(args=None, await_termination=False)
print("   Job submitted!")

print("\n" + "=" * 60)
print("Pipeline launched successfully!")
print("=" * 60)
