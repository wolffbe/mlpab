import os

os.environ["NO_PROXY"] = ""
os.environ["no_proxy"] = ""

import urllib3

urllib3.disable_warnings()

import hopsworks

proj = hopsworks.login(hostname_verification=False)
print("project:", proj.name)

dataset_api = proj.get_dataset_api()
for local in ["data/initial_export.csv", "data/reload/new_export.csv", "ingest_job.py"]:
    path = dataset_api.upload(local, "Resources", overwrite=True)
    print("uploaded:", path)

job_api = proj.get_job_api()
cfg = job_api.get_configuration("PYSPARK")
cfg["appPath"] = "/Projects/{}/Resources/ingest_job.py".format(proj.name)
job = job_api.create_job("customers4baff7_ingest", cfg)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("final status:", execution.final_status, "state:", execution.state)

out, err = execution.download_logs()
print("---- stdout ----")
with open(out) as f:
    print(f.read()[-8000:])
print("---- stderr (tail) ----")
with open(err) as f:
    print(f.read()[-4000:])
