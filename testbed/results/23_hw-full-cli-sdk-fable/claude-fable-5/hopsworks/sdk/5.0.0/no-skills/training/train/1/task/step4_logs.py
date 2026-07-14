import hopsworks

project = hopsworks.login()
js = project.get_job_api()
job = js.get_job("trainjob646af0")
ex = job.get_executions()[0]
print("execution:", ex.id, ex.state, ex.success)
out, err = ex.download_logs()
print("=== STDOUT ===")
print(open(out).read()[-4000:])
print("=== STDERR ===")
print(open(err).read()[-4000:])
