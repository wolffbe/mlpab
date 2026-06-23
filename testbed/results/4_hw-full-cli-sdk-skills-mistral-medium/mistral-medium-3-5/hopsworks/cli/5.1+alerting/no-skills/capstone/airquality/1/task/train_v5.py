#!/usr/bin/env python3
import hopsworks

# Login
hopsworks.login()

# Try to import feature_store directly
try:
    from hopsworks import feature_store
    print(f"Imported feature_store module: {feature_store}")
    fs = feature_store.get_feature_store()
    print(f"Got feature store: {type(fs)}")
except Exception as e:
    print(f"Direct import failed: {e}")

# Try importing from hopsworks.feature_store
try:
    from hopsworks.feature_store import FeatureStore
    fs = FeatureStore()
    print(f"Got FeatureStore via FeatureStore(): {type(fs)}")
except Exception as e:
    print(f"FeatureStore import failed: {e}")

# Try all imports
import sys
import importlib

modules_to_try = [
    "hopsworks.feature_store",
    "hopsworks.feature_store.api",
    "hopsworks.core.feature_store_api",
]

for mod_name in modules_to_try:
    try:
        mod = importlib.import_module(mod_name)
        print(f"Found module: {mod_name}")
        attrs = [x for x in dir(mod) if not x.startswith('_')]
        print(f"  Attributes: {attrs[:10]}...")  # First 10
    except ImportError as e:
        print(f"Cannot import {mod_name}: {e}")

print("Done!")
