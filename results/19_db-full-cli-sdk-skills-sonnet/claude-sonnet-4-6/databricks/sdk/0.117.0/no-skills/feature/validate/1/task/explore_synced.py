#!/usr/bin/env python3
"""Explore Synced Tables API via REST."""

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Try to use the API client directly to explore available endpoints
api = w.api_client

# Check if there's a synced-tables endpoint
print("Checking catalog module for SyncedTable...")
import databricks.sdk.service.catalog as c
synced = [x for x in dir(c) if 'sync' in x.lower() or 'Sync' in x]
print("Synced items in catalog:", synced)

# Check all items
feature_items = [x for x in dir(c) if 'feature' in x.lower() or 'Feature' in x]
print("Feature items in catalog:", feature_items)

# Try to look at the feature_store publish_table method
import inspect
print("\nfeature_store.publish_table signature:")
try:
    print(inspect.signature(w.feature_store.publish_table))
except:
    pass

print("\nhelp(w.feature_store.publish_table):")
try:
    help(w.feature_store.publish_table)
except:
    pass

# Try REST API for synced tables
print("\nTrying REST API for synced tables...")
try:
    resp = api.do("GET", "/api/2.1/unity-catalog/synced-tables")
    print("GET synced-tables response:", resp)
except Exception as e:
    print(f"Error: {e}")
