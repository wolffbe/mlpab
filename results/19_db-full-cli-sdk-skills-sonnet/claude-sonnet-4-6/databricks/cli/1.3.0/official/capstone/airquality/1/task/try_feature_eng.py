# Databricks notebook source

# COMMAND ----------
import json
results = {}

# Check if feature engineering library is available
try:
    import databricks.feature_engineering as fe
    results["fe_available"] = True
    results["fe_attrs"] = [a for a in dir(fe) if not a.startswith("_")]

    fec = fe.FeatureEngineeringClient()
    results["fec_attrs"] = [a for a in dir(fec) if not a.startswith("_")]
except Exception as e:
    results["fe_error"] = str(e)[:200]

try:
    import databricks.feature_store as fs
    results["fs_available"] = True
    fsc = fs.FeatureStoreClient()
    results["fsc_attrs"] = [a for a in dir(fsc) if not a.startswith("_")]
except Exception as e:
    results["fs_error"] = str(e)[:200]

dbutils.notebook.exit(json.dumps(results))
