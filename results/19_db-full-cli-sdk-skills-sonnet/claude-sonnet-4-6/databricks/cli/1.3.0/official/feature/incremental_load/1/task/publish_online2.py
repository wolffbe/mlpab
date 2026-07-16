# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import inspect
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Inspect publish_table signature
sig = inspect.signature(fe.publish_table)
print(f"publish_table signature: {sig}")
print(f"publish_table doc: {fe.publish_table.__doc__}")

# COMMAND ----------

# Explore the module for online store classes
import databricks.feature_engineering as fe_module
fe_attrs = [a for a in dir(fe_module) if 'online' in a.lower() or 'store' in a.lower() or 'publish' in a.lower()]
print("Online-related attributes:", fe_attrs)

# COMMAND ----------

# Check what's in databricks.feature_store
try:
    import databricks.feature_store as fs_module
    fs_attrs = [a for a in dir(fs_module) if 'online' in a.lower() or 'store' in a.lower() or 'spec' in a.lower()]
    print("feature_store online attrs:", fs_attrs)
except Exception as e:
    print(f"feature_store error: {e}")

# COMMAND ----------

# Try to find OnlineStoreSpec or similar
try:
    from databricks.feature_engineering.entities.feature_lookup import OnlineStoreSpec
    print("Found OnlineStoreSpec in feature_lookup")
except Exception as e:
    print(f"OnlineStoreSpec not in feature_lookup: {e}")

try:
    from databricks.feature_engineering import OnlineStoreSpec
    print("Found OnlineStoreSpec in feature_engineering")
except Exception as e:
    print(f"OnlineStoreSpec not in feature_engineering: {e}")

# COMMAND ----------

# Find all entity classes
import pkgutil
import databricks.feature_engineering as pkg
results = {}
for importer, modname, ispkg in pkgutil.walk_packages(path=pkg.__path__, prefix=pkg.__name__+'.'):
    try:
        mod = __import__(modname, fromlist='dummy')
        online_classes = {k: v for k, v in vars(mod).items()
                         if 'online' in k.lower() or 'store' in k.lower() or 'spec' in k.lower()}
        if online_classes:
            results[modname] = list(online_classes.keys())
    except Exception:
        pass

import json
dbutils.notebook.exit(json.dumps({"results": results}))
