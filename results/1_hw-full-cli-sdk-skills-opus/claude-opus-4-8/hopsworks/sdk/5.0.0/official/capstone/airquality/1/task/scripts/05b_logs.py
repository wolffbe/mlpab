import warnings
warnings.filterwarnings("ignore")
import hopsworks
project = hopsworks.login()
api = project.get_jobs_api()
job = api.get_job("airq_infer")
execs = job.get_executions()
ex = execs[-1] if isinstance(execs, list) else execs
out, err = ex.download_logs(".")
print("=== STDOUT ==="); print(open(out).read()[-4000:])
print("=== STDERR ==="); print(open(err).read()[-4000:])
