import os
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)
import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()

ds.mkdir("Resources/scaledaff2b3") if not ds.exists("Resources/scaledaff2b3") else None
for f in ["data/features_train.csv", "data/features_serve.csv"]:
    p = ds.upload(f, "Resources/scaledaff2b3", overwrite=True)
    print("uploaded", p)
script_path = ds.upload("standardize_job.py", "Resources/scaledaff2b3", overwrite=True)
print("uploaded", script_path)

jobs = project.get_job_api()
config = jobs.get_configuration("PYTHON")
config["appPath"] = script_path
job = jobs.create_job("scaledaff2b3_ingest", config)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("final state:", execution.state, execution.final_status)
out, err = execution.get_logs()
print("=== STDOUT ===")
print(out)
print("=== STDERR ===")
print(err)
