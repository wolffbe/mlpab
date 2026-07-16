# Databricks notebook source
import importlib
mods = ["numpy", "pandas", "sklearn", "mlflow", "databricks.feature_engineering",
        "databricks.feature_store", "databricks.sdk", "scipy", "xgboost", "lightgbm"]
stat = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        stat.append("OK:" + m + ":" + str(getattr(mod, "__version__", "?")))
    except Exception as e:
        stat.append("MISSING:" + m)
RESULT = " | ".join(stat)
print(RESULT)

# COMMAND ----------
df = spark.read.option("header", True).option("inferSchema", True).csv("/Volumes/workspace/mlpab2cccc6/raw/airquality_history.csv")
print("rows", df.count())
df.write.mode("overwrite").saveAsTable("workspace.mlpab2cccc6.diag_tmp")
print("write OK")
spark.sql("DROP TABLE IF EXISTS workspace.mlpab2cccc6.diag_tmp")
dbutils.notebook.exit(RESULT + " || write OK rows=" + str(df.count()))
