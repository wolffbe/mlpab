# Databricks notebook source

# COMMAND ----------
import json
results = {}

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# Check what's available in w for synced/online
synced_attrs = [a for a in dir(w) if 'sync' in a.lower() or 'table' in a.lower()]
results["sdk_attrs"] = synced_attrs

import databricks.sdk.service.catalog as c
synced_classes = [a for a in dir(c) if 'sync' in a.lower() or ('table' in a.lower() and 'online' not in a.lower())]
results["catalog_classes"] = synced_classes[:30]

dbutils.notebook.exit(json.dumps(results))
