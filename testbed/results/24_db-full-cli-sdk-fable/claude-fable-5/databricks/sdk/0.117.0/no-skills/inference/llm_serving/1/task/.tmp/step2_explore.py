from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

print("--- clusters ---")
try:
    for c in w.clusters.list():
        print(c.cluster_id, c.cluster_name, c.state)
except Exception as e:
    print("ERR", e)

print("--- serving endpoints ---")
try:
    for e in w.serving_endpoints.list():
        print(e.name, e.state)
except Exception as ex:
    print("ERR", ex)

print("--- volumes API present:", hasattr(w, "volumes"))
print("--- files API present:", hasattr(w, "files"))
print("--- model_registry present:", hasattr(w, "model_registry"))
print("--- registered_models present:", hasattr(w, "registered_models"))
print("--- model_versions present:", hasattr(w, "model_versions"))
print([m for m in dir(w.model_versions) if not m.startswith("_")])
print([m for m in dir(w.registered_models) if not m.startswith("_")])
