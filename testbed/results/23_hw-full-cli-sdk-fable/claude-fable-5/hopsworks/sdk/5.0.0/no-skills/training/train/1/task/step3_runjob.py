import hopsworks

project = hopsworks.login()
js = project.get_job_api()

cfg = js.get_configuration("PYTHON")
cfg["appPath"] = f"/Projects/{project.name}/Resources/trainjob646af0/job_wrapper.py"

job = js.create_job("trainjob646af0", cfg)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("final state:", execution.state, "| success:", execution.success)
