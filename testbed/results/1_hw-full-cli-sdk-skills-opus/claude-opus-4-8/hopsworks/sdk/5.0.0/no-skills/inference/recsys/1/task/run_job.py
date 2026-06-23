import hopsworks

proj = hopsworks.login()
ds = proj.get_dataset_api()
ja = proj.get_jobs_api()

# Upload the compute script to the platform
up = ds.upload("recs_job.py", "Resources", overwrite=True)
print("uploaded to:", up, flush=True)

config = ja.get_configuration("PYTHON")
config["appPath"] = "Resources/recs_job.py"

job = ja.create_job("recsfd473b_compute", config)
print("job created:", job.name, flush=True)

execution = job.run(await_termination=True)
print("final_status:", execution.final_status, "success:", execution.success, flush=True)
print("state:", execution.state, flush=True)

try:
    out, err = execution.download_logs()
    print("LOGS at:", out, err, flush=True)
    for p in [out, err]:
        if p:
            with open(p) as f:
                print(f"===== {p} =====")
                print(f.read()[-4000:])
except Exception as e:
    print("log download err:", e, flush=True)
