# Databricks notebook source
import os, runpy

WORK = "/Volumes/workspace/mlpab5a4ef4/jobdata/work"
os.chdir(WORK)
runpy.run_path(os.path.join(WORK, "train_model.py"), run_name="__main__")
print("script done, predictions.csv exists:", os.path.exists(os.path.join(WORK, "predictions.csv")))

# COMMAND ----------

TABLE = "workspace.mlpab5a4ef4.predictions178367"
df = (
    spark.read.format("csv")
    .option("header", "true")
    .schema("row_id STRING, score DOUBLE")
    .load(f"{WORK}/predictions.csv")
)
df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE)
spark.sql(f"ALTER TABLE {TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
spark.sql(f"ALTER TABLE {TABLE} ALTER COLUMN row_id SET NOT NULL")
spark.sql(f"ALTER TABLE {TABLE} ADD CONSTRAINT predictions178367_pk PRIMARY KEY (row_id)")
print("rows:", spark.table(TABLE).count())
