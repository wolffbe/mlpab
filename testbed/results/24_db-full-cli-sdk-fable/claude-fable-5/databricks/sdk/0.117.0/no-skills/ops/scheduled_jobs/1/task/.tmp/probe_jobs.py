from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
try:
    jobs = list(w.jobs.list(limit=25))
    print("jobs.list OK, count:", len(jobs))
    for j in jobs:
        print(" -", j.job_id, j.settings.name if j.settings else None)
except Exception as e:
    print("jobs.list FAILED:", type(e).__name__, e)
