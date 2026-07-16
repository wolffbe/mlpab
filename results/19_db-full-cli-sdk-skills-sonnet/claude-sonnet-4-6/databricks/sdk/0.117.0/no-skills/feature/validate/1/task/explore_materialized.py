#!/usr/bin/env python3
"""Explore materialized features and different online publishing approaches."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import ml
import inspect

w = WorkspaceClient()

# Look at MaterializedFeature
print("CreateMaterializedFeatureRequest:")
help(ml.CreateMaterializedFeatureRequest)
print()

print("MaterializedFeature:")
help(ml.MaterializedFeature)
print()

print("OnlineStoreConfig:")
help(ml.OnlineStoreConfig)
print()

print("IngestionConfig:")
help(ml.IngestionConfig)
print()

# Check online store that was just created
print("Current online stores:")
stores = list(w.feature_store.list_online_stores())
for s in stores:
    print(f"  {s.name}: state={s.state}")

# Look at PublishTableResponse
print("\nPublishTableResponse:")
help(ml.PublishTableResponse)
