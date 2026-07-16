# Databricks notebook source
# This notebook runs the PROVIDED, UNMODIFIED training script as a job task,
# then loads its predictions output into a feature table with a primary key.

import os

VOLDIR = "/Volumes/workspace/mlpabc9dbc5/trainvol/jobdir"
TABLE = "workspace.mlpabc9dbc5.predictionsa834e5"

# --- Run the provided training script exactly as-is from its working dir ---
os.chdir(VOLDIR)
with open(os.path.join(VOLDIR, "train_model.py")) as f:
    script_src = f.read()
exec(compile(script_src, "train_model.py", "exec"), {"__name__": "__main__"})

print("predictions.csv exists:", os.path.exists(os.path.join(VOLDIR, "predictions.csv")))

# COMMAND ----------

# --- Load the job's predictions output into a feature table (offline Delta) ---
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

pdf = pd.read_csv(os.path.join(VOLDIR, "predictions.csv"))
pdf["row_id"] = pdf["row_id"].astype(str)
pdf["score"] = pdf["score"].astype(float)

schema = StructType([
    StructField("row_id", StringType(), False),
    StructField("score", DoubleType(), True),
])
sdf = spark.createDataFrame(pdf[["row_id", "score"]], schema=schema)
sdf.createOrReplaceTempView("preds_tmp")

spark.sql(f"DROP TABLE IF EXISTS {TABLE}")
spark.sql(f"""
CREATE TABLE {TABLE} (
  row_id STRING NOT NULL,
  score DOUBLE,
  CONSTRAINT predictionsa834e5_pk PRIMARY KEY (row_id)
)
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
spark.sql(f"INSERT INTO {TABLE} SELECT row_id, score FROM preds_tmp")

cnt = spark.table(TABLE).count()
print("rows in feature table:", cnt)
display(spark.table(TABLE).limit(5))
