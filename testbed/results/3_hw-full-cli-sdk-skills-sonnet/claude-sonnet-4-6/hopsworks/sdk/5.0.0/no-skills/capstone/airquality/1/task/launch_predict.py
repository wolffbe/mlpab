"""Launch predict_script.py as a Hopsworks Job."""
import hopsworks

project = hopsworks.login()

dataset_api = project.get_dataset_api()
print("Uploading predict_script.py...")
dataset_api.upload("predict_script.py", "Resources", overwrite=True)

jobs_api = project.get_jobs_api()
job_name = "airq_predict_job"

job_config = jobs_api.get_configuration("PYTHON")
job_config["appPath"] = "hdfs:///Projects/{}/Resources/predict_script.py".format(project.name)

job = jobs_api.get_job(job_name)
if job is None:
    job = jobs_api.create_job(job_name, job_config)
    print("Job created.")
else:
    print("Job exists.")

print("Running prediction job...")
execution = job.run(await_termination=True)
print(f"Execution final_status: {execution.final_status}")
print(f"Success: {execution.success}")

if not execution.success:
    log_files = execution.download_logs()
    stdout_path, stderr_path = log_files
    print("=== STDOUT ===")
    with open(stdout_path) as f:
        print(f.read())
    print("=== STDERR (last 3000 chars) ===")
    with open(stderr_path) as f:
        content = f.read()
        print(content[-3000:])
