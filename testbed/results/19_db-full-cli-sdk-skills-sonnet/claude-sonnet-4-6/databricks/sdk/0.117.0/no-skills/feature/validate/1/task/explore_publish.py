#!/usr/bin/env python3
"""Explore publish_table and feature serving options."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import ml
import inspect

w = WorkspaceClient()

# Explore PublishSpec
print("PublishSpec:")
help(ml.PublishSpec)
print()

# Look for online store classes
online_items = [x for x in dir(ml) if 'online' in x.lower() or 'Online' in x or 'store' in x.lower()]
print("Online/Store items in ml module:", online_items)

# Get full list of ml module members
print("\nAll ml module classes:")
for name in sorted(dir(ml)):
    if not name.startswith('_'):
        obj = getattr(ml, name)
        if isinstance(obj, type):
            print(f"  {name}")
