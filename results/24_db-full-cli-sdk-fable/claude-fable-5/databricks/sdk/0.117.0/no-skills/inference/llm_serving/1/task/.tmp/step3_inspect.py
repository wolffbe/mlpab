from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

ep = w.serving_endpoints.get("scorer40bb09")
print("config:", ep.config)
print("pending:", ep.pending_config)
print("state:", ep.state)

print("--- UC registered models (all schemas) ---")
try:
    for m in w.registered_models.list(max_results=50):
        print(m.full_name)
except Exception as e:
    print("ERR", e)
