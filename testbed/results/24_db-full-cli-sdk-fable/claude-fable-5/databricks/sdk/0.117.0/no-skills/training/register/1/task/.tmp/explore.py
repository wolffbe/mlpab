from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print("me:", w.current_user.me().user_name)
print([a for a in dir(w) if 'model' in a.lower() or 'registr' in a.lower()])
print("registered_models:", [m for m in dir(w.registered_models) if not m.startswith('_')])
print("model_versions:", [m for m in dir(w.model_versions) if not m.startswith('_')])
print("model_registry:", [m for m in dir(w.model_registry) if not m.startswith('_')])
print("files:", [m for m in dir(w.files) if not m.startswith('_')])
