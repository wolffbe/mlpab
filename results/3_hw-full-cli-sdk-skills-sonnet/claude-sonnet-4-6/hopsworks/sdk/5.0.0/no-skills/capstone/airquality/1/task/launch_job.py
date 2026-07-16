"""Launch train_script.py as a Hopsworks Job (training runs on the platform)."""
import hopsworks
import os

project = hopsworks.login()

# Upload the training script to the dataset
dataset_api = project.get_dataset_api()

print("Uploading train_script.py to platform...")
dataset_api.upload("train_script.py", "Resources", overwrite=True)

# Create and run the job
jobs_api = project.get_jobs_api()

job_name = "airq_train_job"
job_config = jobs_api.get_configuration("PYTHON")
job_config["appPath"] = "hdfs:///Projects/{}/Resources/train_script.py".format(project.name)

print(f"Creating job {job_name}...")
job = jobs_api.get_job(job_name)
if job is None:
    job = jobs_api.create_job(job_name, job_config)
    print("Job created.")
else:
    print("Job exists.")

print("Starting job execution...")
execution = job.run(await_termination=True)

print(f"Execution state: {execution.final_status}")
print(f"Execution success: {execution.success}")

if not execution.success:
    print("Job failed! Fetching logs...")
    try:
        logs = execution.get_logs()
        print(logs)
    except Exception as e:
        print(f"Could not get logs: {e}")
else:
    print("Training job completed successfully!")
