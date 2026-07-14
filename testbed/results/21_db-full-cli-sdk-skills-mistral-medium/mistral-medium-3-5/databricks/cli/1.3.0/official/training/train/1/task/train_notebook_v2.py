# Databricks notebook source
# MAGIC %python
# MAGIC import os
# MAGIC import shutil
# MAGIC import subprocess
# MAGIC import sys
# MAGIC 
# MAGIC # In Databricks, workspace files are accessible via /Workspace
# MAGIC workspace_path = "/Workspace/Users/benedict@logicalclocks.com/mlpab155832/data"
# MAGIC current_dir = os.getcwd()
# MAGIC 
# MAGIC print(f"Current working directory: {current_dir}")
# MAGIC print(f"Copying files from: {workspace_path}")
# MAGIC 
# MAGIC # Copy the data files
# MAGIC files_to_copy = ["train.csv", "score.csv", "train_model.py"]
# MAGIC for filename in files_to_copy:
# MAGIC     src = os.path.join(workspace_path, filename)
# MAGIC     dst = os.path.join(current_dir, filename)
# MAGIC     print(f"Copying {src} to {dst}")
# MAGIC     if os.path.exists(src):
# MAGIC         shutil.copy2(src, dst)
# MAGIC     else:
# MAGIC         print(f"Source file {src} does not exist!")
# MAGIC 
# MAGIC # Run the training script
# MAGIC if os.path.exists("train_model.py"):
# MAGIC     print("Running training script...")
# MAGIC     result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
# MAGIC     print("STDOUT:", result.stdout)
# MAGIC     print("STDERR:", result.stderr)
# MAGIC     print("Return code:", result.returncode)
# MAGIC 
# MAGIC     # Copy predictions back to workspace
# MAGIC     predictions_src = os.path.join(current_dir, "predictions.csv")
# MAGIC     predictions_dst = os.path.join(workspace_path, "predictions.csv")
# MAGIC     if os.path.exists(predictions_src):
# MAGIC         print(f"Copying predictions.csv to {predictions_dst}")
# MAGIC         shutil.copy2(predictions_src, predictions_dst)
# MAGIC     else:
# MAGIC         print("predictions.csv not found!")
# MAGIC         sys.exit(1)
# MAGIC else:
# MAGIC     print("train_model.py not found in current directory!")
# MAGIC     sys.exit(1)