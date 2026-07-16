import sys
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print("clusters:", flush=True)
try:
    for c in w.clusters.list():
        print(" ", c.cluster_id, c.cluster_name, c.state, c.cluster_source, flush=True)
except Exception as e:
    print("  err:", e, flush=True)

print("existing jobs:", flush=True)
try:
    for j in w.jobs.list(limit=5):
        print(" ", j.job_id, j.settings.name if j.settings else None, flush=True)
except Exception as e:
    print("  err:", e, flush=True)

print("node types:", flush=True)
try:
    nts = w.clusters.list_node_types()
    print("  count:", len(nts.node_types or []), flush=True)
    for nt in (nts.node_types or [])[:3]:
        print(" ", nt.node_type_id, flush=True)
except Exception as e:
    print("  err:", type(e).__name__, e, flush=True)
