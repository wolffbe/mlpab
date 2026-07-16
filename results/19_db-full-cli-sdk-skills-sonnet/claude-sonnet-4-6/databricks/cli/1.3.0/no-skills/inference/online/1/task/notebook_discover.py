# Databricks notebook source
import json
import requests
import importlib, pkgutil

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"
prefix = "mlpabcef85c"

results = {}

# COMMAND ----------
# Check Feature Engineering SDK
print("=== Feature Engineering SDK ===")
try:
    import databricks.feature_engineering as fe
    print(f"Feature Engineering module: {dir(fe)}")
    results["fe_available"] = True
except ImportError as e:
    print(f"Not available: {e}")
    results["fe_available"] = False

# COMMAND ----------
# Check databricks.sdk
print("=== Databricks SDK ===")
try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    print("WorkspaceClient created")
    print(f"SDK services: {[s for s in dir(w) if not s.startswith('_')]}")
    results["sdk_available"] = True
    results["sdk_services"] = [s for s in dir(w) if not s.startswith('_')]
except ImportError as e:
    print(f"SDK not available: {e}")
    results["sdk_available"] = False

# COMMAND ----------
# Check if the SDK has synced_tables or online_tables
if results.get("sdk_available"):
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    # Check for online tables related services
    for attr in ["online_tables", "synced_tables", "feature_engineering", "feature_serving", "feature_store"]:
        has = hasattr(w, attr)
        print(f"w.{attr}: {has}")
        results[f"sdk_{attr}"] = has

# COMMAND ----------
# Check Lakebase API
print("=== Lakebase API ===")
for path in [
    "/api/2.0/lakebase",
    "/api/2.1/lakebase",
    "/api/2.0/database",
    "/api/2.0/databases",
    "/api/2.1/databases",
]:
    r = requests.get(f"{host}{path}", headers=headers)
    print(f"GET {path}: {r.status_code} {r.text[:200]}")
    results[f"lakebase_{path}"] = r.status_code

# COMMAND ----------
# Try feature engineering REST API
print("=== Feature Engineering REST API ===")
for path in [
    "/api/2.0/feature-engineering",
    "/api/2.0/feature-engineering/feature-tables",
    "/api/2.0/feature-engineering/online-feature-tables",
    "/api/2.1/feature-engineering",
]:
    r = requests.get(f"{host}{path}", headers=headers)
    print(f"GET {path}: {r.status_code} {r.text[:300]}")
    results[f"fe_{path}"] = r.status_code

# COMMAND ----------
# Try the databricks SDK synced tables directly
print("=== SDK Synced Tables ===")
try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    if hasattr(w, 'online_tables'):
        print(f"online_tables methods: {dir(w.online_tables)}")
        # Try to create an online table via SDK
        try:
            from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
            spec = OnlineTableSpec(
                source_table_full_name=full_table_name,
                primary_key_columns=["account_id"],
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
            )
            print(f"Online table spec created: {spec}")
            # Try to create
            result = w.online_tables.create(name=full_table_name, spec=spec)
            print(f"Create result: {result}")
            results["sdk_create_online"] = str(result)
        except Exception as e:
            print(f"SDK create error: {e}")
            results["sdk_create_online_error"] = str(e)
    else:
        print("No online_tables service in SDK")
except Exception as e:
    print(f"SDK error: {e}")

# COMMAND ----------
print(json.dumps(results, indent=2))
dbutils.notebook.exit(json.dumps(results))
