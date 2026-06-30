import datetime
import databricks.sdk as dsdk

w = dsdk.WorkspaceClient()
JOB_NAME = "trainjoba834e5"
job = list(w.jobs.list(name=JOB_NAME))[0]
print("running job", job.job_id)
run = w.jobs.run_now_and_wait(job_id=job.job_id,
                              timeout=datetime.timedelta(minutes=30))
print("life_cycle:", run.state.life_cycle_state)
print("result:", run.state.result_state)
print("message:", run.state.state_message)
print("run_id:", run.run_id)
# fetch task output
for t in run.tasks or []:
    try:
        out = w.jobs.get_run_output(run_id=t.run_id)
        print("--- task", t.task_key, "notebook output ---")
        print(getattr(out, "logs", None))
        print(out.notebook_output)
        if out.error:
            print("ERROR:", out.error)
    except Exception as e:
        print("output err:", repr(e)[:300])
