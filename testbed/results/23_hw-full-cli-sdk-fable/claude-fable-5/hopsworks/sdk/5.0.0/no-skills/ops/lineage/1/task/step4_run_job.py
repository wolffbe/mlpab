import hopsworks

project = hopsworks.login()
print("project:", project.name)

dataset_api = project.get_dataset_api()
for f in ["data/raw_a.csv", "data/raw_b.csv", "platform_etl_8af783.py"]:
    path = dataset_api.upload(f, "Resources", overwrite=True)
    print("uploaded:", path)

job_api = project.get_job_api()
config = job_api.get_configuration("PYSPARK")
config["appPath"] = "/Projects/" + project.name + "/Resources/platform_etl_8af783.py"
job = job_api.create_job("etl8af783", config)

execution = job.run(await_termination=True)
print("state:", execution.state, "final_status:", execution.final_status)
out, err = execution.download_logs()
print("logs:", out, err)
