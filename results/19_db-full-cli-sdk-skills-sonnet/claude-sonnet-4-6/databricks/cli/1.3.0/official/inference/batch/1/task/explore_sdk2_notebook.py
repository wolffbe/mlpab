# Databricks notebook source
# COMMAND ----------
import inspect
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as catalog_svc

w = WorkspaceClient()

# Inspect the create method signature
sig = inspect.signature(w.online_tables.create)
print(f"create signature: {sig}")

# Also check the OnlineTable class
print(f"\nOnlineTable fields: {[f for f in dir(catalog_svc.OnlineTable) if not f.startswith('_')]}")
print(f"\nOnlineTableSpec fields: {[f for f in dir(catalog_svc.OnlineTableSpec) if not f.startswith('_')]}")

# Try to check synced tables via another approach - maybe it's feature store
try:
    from databricks.sdk.service import iam
    feature_store = [attr for attr in dir(w) if 'feature' in attr.lower()]
    print(f"\nFeature store attrs: {feature_store}")
except:
    pass

dbutils.notebook.exit("explored")
