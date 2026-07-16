# Databricks notebook source
# COMMAND ----------
import json
import subprocess
import pkg_resources

results = {}

# Check installed SDK version
try:
    sdk_version = pkg_resources.get_distribution("databricks-sdk").version
    results["databricks-sdk"] = sdk_version
except:
    pass

try:
    fe_version = pkg_resources.get_distribution("databricks-feature-engineering").version
    results["databricks-feature-engineering"] = fe_version
except:
    results["databricks-feature-engineering"] = "not installed"

try:
    fs_version = pkg_resources.get_distribution("databricks-feature-store").version
    results["databricks-feature-store"] = fs_version
except:
    results["databricks-feature-store"] = "not installed"

# Check all databricks packages
databricks_pkgs = {pkg.key: pkg.version for pkg in pkg_resources.working_set if 'databricks' in pkg.key.lower()}
results["all_databricks_pkgs"] = databricks_pkgs

# Check if w.synced_tables exists
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
all_w_attrs = [a for a in dir(w) if not a.startswith('_')]
results["all_w_attrs"] = sorted(all_w_attrs)

dbutils.notebook.exit(json.dumps(results))
