# Databricks notebook source
# COMMAND ----------
import inspect
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as catalog_svc

w = WorkspaceClient()

lines = []

# Inspect the create method signature
sig = inspect.signature(w.online_tables.create)
lines.append(f"create sig: {sig}")

# Check what OnlineTable looks like
try:
    ot_init = inspect.signature(catalog_svc.OnlineTable.__init__)
    lines.append(f"OnlineTable init: {ot_init}")
except Exception as e:
    lines.append(f"OnlineTable init error: {e}")

try:
    ots_init = inspect.signature(catalog_svc.OnlineTableSpec.__init__)
    lines.append(f"OnlineTableSpec init: {ots_init}")
except Exception as e:
    lines.append(f"OnlineTableSpec error: {e}")

# Check for synced tables in catalog service
synced = [attr for attr in dir(catalog_svc) if 'synced' in attr.lower() or 'Synced' in attr]
lines.append(f"Synced in catalog: {synced}")

# pip show databricks-sdk
import subprocess
try:
    result = subprocess.run(['pip', 'show', 'databricks-sdk'], capture_output=True, text=True)
    lines.append(f"SDK info: {result.stdout[:200]}")
except:
    lines.append("pip failed")

dbutils.notebook.exit("\n".join(lines))
