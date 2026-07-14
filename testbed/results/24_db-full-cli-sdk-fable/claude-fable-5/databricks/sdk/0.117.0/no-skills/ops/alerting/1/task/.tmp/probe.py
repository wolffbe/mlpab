from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

try:
    js = list(w.jobs.list(limit=5))
    print("jobs.list OK, count:", len(js))
    for j in js:
        print(" ", j.job_id, j.settings.name if j.settings else None)
except Exception as e:
    print("jobs.list FAIL:", e)

try:
    cs = list(w.clusters.list())
    print("clusters.list OK, count:", len(cs))
except Exception as e:
    print("clusters.list FAIL:", type(e).__name__, str(e)[:120])

try:
    ws = list(w.warehouses.list())
    print("warehouses.list OK:", [(x.id, x.name, str(x.state)) for x in ws])
except Exception as e:
    print("warehouses.list FAIL:", type(e).__name__, str(e)[:120])
