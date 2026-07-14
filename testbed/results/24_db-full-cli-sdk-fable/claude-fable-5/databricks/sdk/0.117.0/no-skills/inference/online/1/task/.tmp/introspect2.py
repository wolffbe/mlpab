import databricks.sdk

w = databricks.sdk.WorkspaceClient()
for svc in ["feature_store", "feature_engineering", "online_tables", "database", "postgres", "serving_endpoints", "statement_execution", "warehouses"]:
    obj = getattr(w, svc)
    print(svc, "->", [m for m in dir(obj) if not m.startswith("_")])
    print()
