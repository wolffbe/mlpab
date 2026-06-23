import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
ds = project.get_dataset_api()
proj = project.name

def ensure_dir(path):
    mk = getattr(ds, "mkdir", None)
    if mk:
        try:
            mk(path)
        except Exception as e:
            print("mkdir", path, "->", repr(e)[:120])

data_dir = "Resources/recs_data"
job_dir = "Resources/jobs/recsfd473b"
ensure_dir(data_dir)
ensure_dir(job_dir)

for fn in ["user_embeddings.csv", "item_embeddings.csv", "interactions.csv"]:
    print("upload", fn, "->", ds.upload(f"data/{fn}", data_dir, overwrite=True))
print("upload script ->", ds.upload("recs_job.py", job_dir, overwrite=True))

api = project.get_job_api()
cfg = api.get_configuration("PYTHON")
cfg["appPath"] = f"/Projects/{proj}/{job_dir}/recs_job.py"
print("env", cfg["environmentName"], "appPath", cfg["appPath"])

job = api.create_job(name="recsfd473b_job", config=cfg)
print("created job", job.name)

execution = job.run(await_termination=True)
print("execution state", execution.state, "final status", execution.final_status)
