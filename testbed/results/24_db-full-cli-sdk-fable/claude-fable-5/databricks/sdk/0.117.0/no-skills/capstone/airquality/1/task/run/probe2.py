import inspect
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import pipelines

w = WorkspaceClient()
sig = inspect.signature(w.pipelines.create)
for p in sig.parameters.values():
    print(p.name, flush=True)
print("---", flush=True)
print([x for x in dir(pipelines) if "Environment" in x or "Library" in x], flush=True)
print("--- existing pipelines:", flush=True)
try:
    for pl in w.pipelines.list_pipelines(max_results=10):
        print(" ", pl.pipeline_id, pl.name, pl.state, flush=True)
except Exception as e:
    print("  err:", e, flush=True)
print("--- apps:", flush=True)
try:
    for a in w.apps.list():
        print(" ", a.name, a.compute_status, flush=True)
    print("apps listable", flush=True)
except Exception as e:
    print("  err:", e, flush=True)
