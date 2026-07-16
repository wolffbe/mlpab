# Databricks notebook source
res = []
try:
    import mlflow
    res.append("mlflow:" + mlflow.__version__)
except Exception as e:
    res.append("NO mlflow:" + repr(e)[:80])
try:
    import databricks.feature_engineering as fe
    res.append("fe:OK")
except Exception as e:
    res.append("NO fe:" + repr(e)[:80])
dbutils.notebook.exit(" | ".join(res))
