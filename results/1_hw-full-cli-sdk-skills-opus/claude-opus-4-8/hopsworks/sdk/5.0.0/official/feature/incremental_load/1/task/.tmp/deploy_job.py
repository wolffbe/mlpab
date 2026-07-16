import hopsworks

project = hopsworks.login()
api = project.get_job_api()
dataset_api = project.get_dataset_api()

JOB_NAME = "incrementaljob614551"
local_script = ".tmp/incremental_pipeline.py"
hdfs_dir = "Resources/jobs/incrementaljob614551"

# Upload script to HopsFS
try:
    dataset_api.mkdir(hdfs_dir)
except Exception as e:
    print("mkdir:", e)
uploaded = dataset_api.upload(local_script, hdfs_dir, overwrite=True)
print("uploaded ->", uploaded)

# Build PYTHON job configuration
config = api.get_configuration("PYTHON")
config["appPath"] = f"/Projects/{project.name}/{hdfs_dir}/incremental_pipeline.py"
print("config keys:", list(config.keys()))

job = api.create_job(name=JOB_NAME, config=config)
print("Created job:", job.name, "id:", job.id)

# Attach a daily recurring schedule
print("schedule_job_save attrs:", [m for m in dir(job) if "schedule" in m.lower()])
