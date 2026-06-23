import inspect, hopsworks
project = hopsworks.login()
api = project.get_job_api()
job = api.get_job("recsfd473b_job")
execs = job.get_executions()
ex = execs[-1] if execs else None
print("methods", [m for m in dir(ex) if not m.startswith("_")])
# try download logs
for m in ["download_logs", "get_logs"]:
    fn = getattr(ex, m, None)
    if fn:
        try:
            print(f"--- {m} sig", inspect.signature(fn))
        except Exception as e:
            print(m, "sig err", e)
try:
    out = ex.download_logs(".")
    print("downloaded logs to", out)
except Exception as e:
    print("download_logs err", repr(e))
