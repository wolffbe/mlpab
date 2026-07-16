# Databricks notebook source
# COMMAND ----------
import subprocess
import shutil
import os

# COMMAND ----------
# Set up working directory
work_dir = "/tmp/trainjob7b586d"
os.makedirs(work_dir, exist_ok=True)

# Copy files from volume to working directory
volume_path = "/Volumes/workspace/mlpab5c18ba/mlpab5c18ba_data"
shutil.copy(f"{volume_path}/train.csv", f"{work_dir}/train.csv")
shutil.copy(f"{volume_path}/score.csv", f"{work_dir}/score.csv")
shutil.copy(f"{volume_path}/train_model.py", f"{work_dir}/train_model.py")

print("Files copied successfully")

# COMMAND ----------
# Run the training script
os.chdir(work_dir)
import subprocess
result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True, cwd=work_dir)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
if result.returncode != 0:
    raise RuntimeError(f"Training script failed: {result.stderr}")

# COMMAND ----------
# Verify predictions.csv was created
import pandas as pd
preds = pd.read_csv(f"{work_dir}/predictions.csv")
print(f"Predictions shape: {preds.shape}")
print(preds.head())

# COMMAND ----------
# Save predictions.csv back to volume
shutil.copy(f"{work_dir}/predictions.csv", f"{volume_path}/predictions.csv")
print("Predictions saved to volume")

# COMMAND ----------
# Also save predictions to a Delta table for feature store
spark.createDataFrame(preds).write.mode("overwrite").saveAsTable("workspace.mlpab5c18ba.predictions7b586d")
print("Predictions saved to Delta table")
