# Databricks notebook source
# COMMAND ----------
import json

results = {}

# Check if databricks feature engineering client is available
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    results['fe_client_available'] = True
    results['fe_methods'] = [m for m in dir(fe) if not m.startswith('_')]
except ImportError as e:
    results['fe_client_error'] = str(e)

# Also check databricks.feature_store
try:
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    results['fs_client_available'] = True
    results['fs_methods'] = [m for m in dir(fs) if not m.startswith('_')]
except ImportError as e:
    results['fs_client_error'] = str(e)

# Check databricks.vector_search
try:
    import databricks.vectordb
    results['vectordb_available'] = True
except ImportError:
    results['vectordb_available'] = False

dbutils.notebook.exit(json.dumps(results))
