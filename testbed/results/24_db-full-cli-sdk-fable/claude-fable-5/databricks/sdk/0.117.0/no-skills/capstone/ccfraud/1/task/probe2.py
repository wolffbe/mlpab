from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print("clusters:")
try:
    for c in w.clusters.list():
        print(" ", c.cluster_id, c.cluster_name, c.state, c.cluster_source)
except Exception as e:
    print("  err:", e)

print("retry submit probe:")
try:
    from databricks.sdk.service import jobs
    r = w.jobs.submit(
        run_name="mlpab67db84_probe",
        tasks=[jobs.SubmitTask(task_key="t", notebook_task=jobs.NotebookTask(notebook_path="/Users/benedict@hopsworks.ai/mlpab67db84/ccfraud_pipeline"))],
    )
    print("  submitted run", r.run_id)
    w.jobs.cancel_run(run_id=r.response.run_id) if hasattr(r, "response") else None
except Exception as e:
    print("  err:", e)

print("spark versions (can we create clusters?):")
try:
    sv = w.clusters.spark_versions()
    print("  ok,", len(sv.versions), "versions")
except Exception as e:
    print("  err:", e)
