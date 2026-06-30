import databricks.sdk as dbsdk

w = dbsdk.WorkspaceClient()
for j in w.jobs.list(expand_tasks=True):
    if "flaky62cc43" in (j.settings.name or ""):
        print("JOB", j.job_id, "NAME", j.settings.name)
        print("  job_email_on_failure:", j.settings.email_notifications)
        print("  job_webhook:", j.settings.webhook_notifications)
        for t in (j.settings.tasks or []):
            nb = t.notebook_task.notebook_path if t.notebook_task else None
            sp = t.spark_python_task.python_file if t.spark_python_task else None
            print("  task", t.task_key, "nb:", nb, "spark:", sp)
            print("    task_email:", t.email_notifications)
        for r in w.jobs.list_runs(job_id=j.job_id):
            print("  RUN", r.run_id, "life:", r.state.life_cycle_state, "result:", r.state.result_state, "msg:", r.state.state_message)
