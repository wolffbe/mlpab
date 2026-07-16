import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
ds.upload("scripts/train_infer.py", "Resources/airq", overwrite=True)
app_path = f"/Projects/{project.name}/Resources/airq/train_infer.py"
print("appPath:", app_path)

api = project.get_jobs_api()
config = api.get_configuration("PYTHON")
config["appPath"] = app_path
config["environmentName"] = "pandas-training-pipeline"

job = api.create_job(name="airq_train_infer", config=config)
print("created job:", job.name)

execution = job.run(await_termination=True)
print("execution state:", execution.state, "final status:", execution.final_status)
print("success:", getattr(execution, "success", None))
