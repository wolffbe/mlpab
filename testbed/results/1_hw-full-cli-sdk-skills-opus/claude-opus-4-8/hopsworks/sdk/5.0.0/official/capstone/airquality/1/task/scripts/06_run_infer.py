import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
ds.upload("scripts/infer.py", "Resources/airq", overwrite=True)
app_path = f"/Projects/{project.name}/Resources/airq/infer.py"

api = project.get_jobs_api()
config = api.get_configuration("PYTHON")
config["appPath"] = app_path
config["environmentName"] = "pandas-training-pipeline"
job = api.create_job(name="airq_infer", config=config)
print("created job:", job.name)
execution = job.run(await_termination=True)
print("state:", execution.state, "final:", execution.final_status)
