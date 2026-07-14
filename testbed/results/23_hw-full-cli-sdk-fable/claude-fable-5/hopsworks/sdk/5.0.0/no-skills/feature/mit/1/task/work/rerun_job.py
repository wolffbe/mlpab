import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()
job_api = project.get_job_api()

folder = "Resources/featureseb4964"
script_path = dataset_api.upload("work/fg_job.py", folder, overwrite=True)
print("uploaded script to:", script_path)

job = job_api.get_job("featureseb4964_build")
execution = job.run(await_termination=True)
print("final status:", execution.final_status, "| state:", execution.state)
