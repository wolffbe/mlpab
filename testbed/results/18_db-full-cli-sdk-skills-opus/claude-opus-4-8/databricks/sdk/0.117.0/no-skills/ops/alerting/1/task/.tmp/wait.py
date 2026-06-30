import databricks.sdk as dbsdk

w = dbsdk.WorkspaceClient()
RUN_ID = 1091041924030040
run = w.jobs.get_run(run_id=RUN_ID)
for t in (run.tasks or []):
    print("TASK", t.task_key, t.state.result_state)
    try:
        ro = w.jobs.get_run_output(run_id=t.run_id)
        print("  ERROR:", ro.error)
        print("  TRACE tail:", (ro.error_trace or "")[-600:])
    except Exception as e:
        print("  no output:", e)
