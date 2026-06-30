# Databricks notebook source
import os
import shutil
import subprocess

# Volume path (accessible from Python and Spark)
vol_path = "/Volumes/workspace/mlpab6c8eeb/mlpab6c8eeb_data"
work_dir = "/tmp/trainjob7b586d"
os.makedirs(work_dir, exist_ok=True)

# Copy input files from volume to local temp dir for the Python training script
shutil.copy(f"{vol_path}/train.csv", f"{work_dir}/train.csv")
shutil.copy(f"{vol_path}/score.csv", f"{work_dir}/score.csv")
shutil.copy(f"{vol_path}/train_model.py", f"{work_dir}/train_model.py")

# Run training script from work_dir (it reads/writes relative to cwd)
result = subprocess.run(
    ["python", "train_model.py"],
    cwd=work_dir,
    capture_output=True,
    text=True
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
if result.returncode != 0:
    raise RuntimeError(f"Training script failed: {result.stderr}")

# Copy predictions.csv to volume for persistence
shutil.copy(f"{work_dir}/predictions.csv", f"{vol_path}/predictions.csv")
print("predictions.csv saved to volume")

# Read predictions from volume (Spark can access /Volumes paths)
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv(f"{vol_path}/predictions.csv", header=True, inferSchema=True)
df.show(5)

# Write to Delta table in the schema
df.write.format("delta").mode("overwrite").saveAsTable("workspace.mlpab6c8eeb.predictions7b586d")
print("Delta table predictions7b586d created successfully")
