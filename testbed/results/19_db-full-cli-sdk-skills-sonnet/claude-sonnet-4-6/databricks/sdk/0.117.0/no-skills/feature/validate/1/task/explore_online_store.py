#!/usr/bin/env python3
"""Explore online store and feature store publishing."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import ml
import inspect

w = WorkspaceClient()

# Check OnlineStore class
print("OnlineStore:")
help(ml.OnlineStore)
print()

print("OnlineStoreConfig:")
help(ml.OnlineStoreConfig)
print()

# Check existing online stores
print("Listing existing online stores:")
try:
    stores = list(w.feature_store.list_online_stores())
    print(f"  Found {len(stores)} stores")
    for s in stores:
        print(f"  Store: {s}")
except Exception as e:
    print(f"  Error: {e}")

# Check create_online_store signature
print("\ncreate_online_store signature:")
help(w.feature_store.create_online_store)

print("\nPublishSpecPublishMode:")
help(ml.PublishSpecPublishMode)
