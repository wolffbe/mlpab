#!/usr/bin/env python3
"""Explore SDK for Synced Tables and online access alternatives."""

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Check for synced tables
print("Checking materialized_features:", dir(w.materialized_features))
print()

# Check feature_store more
print("feature_store methods:")
for m in dir(w.feature_store):
    if not m.startswith('_'):
        print(f"  {m}")
print()

# Check feature_engineering
print("feature_engineering methods:")
for m in dir(w.feature_engineering):
    if not m.startswith('_'):
        print(f"  {m}")
print()

# Check if there are synced_tables or serving_endpoints for feature serving
print("Checking if there's a synced tables in catalog APIs...")
import databricks.sdk.service.catalog as catalog_svc
members = [x for x in dir(catalog_svc) if 'sync' in x.lower() or 'online' in x.lower() or 'feature' in x.lower()]
print("Catalog service members:", members)

# Check all services
import databricks.sdk.service as svc
print("\nAll SDK services:", [x for x in dir(svc) if not x.startswith('_')])
