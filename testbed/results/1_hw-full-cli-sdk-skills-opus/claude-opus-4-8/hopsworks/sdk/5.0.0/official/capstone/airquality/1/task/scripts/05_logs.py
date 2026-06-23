import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
api = project.get_jobs_api()
job = api.get_job("airq_train_infer")
execs = job.get_executions()
ex = execs[-1] if isinstance(execs, list) else execs
print("EXEC methods:", [m for m in dir(ex) if not m.startswith("_")])
print("state:", ex.state, "final:", ex.final_status)
try:
    out, err = ex.download_logs(".")
    print("OUT LOG:", out)
    print("ERR LOG:", err)
    for f in (out, err):
        try:
            print(f"===== {f} =====")
            print(open(f).read()[-6000:])
        except Exception as e:
            print("read err", repr(e))
except Exception as e:
    print("download_logs err:", repr(e))
