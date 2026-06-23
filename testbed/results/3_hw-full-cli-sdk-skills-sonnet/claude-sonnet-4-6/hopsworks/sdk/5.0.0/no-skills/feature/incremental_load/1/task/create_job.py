import hopsworks
import json

project = hopsworks.login()

print("Connected to Hopsworks")

# Upload the ingestion script to Hopsworks dataset
dataset_api = project.get_dataset_api()

# Upload the script
dataset_api.upload("incremental_ingest.py", "Resources", overwrite=True)
print("Uploaded incremental_ingest.py to Resources")

# Create the job
jobs_api = project.get_jobs_api()

# Get PYSPARK config and configure it
config = jobs_api.get_configuration("PYSPARK")
config["appPath"] = "hdfs:///Projects/" + project.name + "/Resources/incremental_ingest.py"

print(f"Creating job with config: {json.dumps(config, indent=2)}")

job = jobs_api.create_job("incrementaljob811051", config)
print(f"Job created: {job.name}")

# Now set up the schedule
print(dir(job))
help(job.schedule)
