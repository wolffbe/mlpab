import hopsworks

project = hopsworks.login()
job_api = project.get_job_api()
for jname in [
    "airqpred754fa9_1_offline_fg_materialization",
    "airq754fa9_1_offline_fg_materialization",
]:
    try:
        job = job_api.get_job(jname)
        exes = job.get_executions()
        for e in sorted(exes, key=lambda x: x.id):
            print(jname, "exec", e.id, "state:", e.state, "final:", e.final_status)
    except Exception as e:  # noqa: BLE001
        print(jname, "error:", e)
