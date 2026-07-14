import hw_env  # noqa: F401
import hopsworks

proj = hopsworks.login()
ja = proj.get_job_api()
job = ja.get_job("drift_investigation")
ex = ja.last_execution(job)
if isinstance(ex, list):
    ex = ex[0]
print("execution", ex.id, ex.state, ex.final_status)
out_path, err_path = ex.download_logs()
for p in (out_path, err_path):
    print("=====", p, "=====")
    try:
        with open(p) as fh:
            t = fh.read()
        print(t[-6000:])
    except Exception as e:
        print("read failed:", e)
