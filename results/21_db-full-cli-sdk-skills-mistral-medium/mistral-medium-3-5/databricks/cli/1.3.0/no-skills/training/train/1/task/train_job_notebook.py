# Databricks notebook source
# MAGIC %python
# MAGIC import shutil
# MAGIC import os
# MAGIC 
# MAGIC # Copy the data files to the current working directory
# MAGIC shutil.copy("/Workspace/${MLPAB_DATABRICKS_PREFIX}_train.csv", "train.csv")
# MAGIC shutil.copy("/Workspace/${MLPAB_DATABRICKS_PREFIX}_score.csv", "score.csv")
# MAGIC shutil.copy("/Workspace/${MLPAB_DATABRICKS_PREFIX}_train_notebook", "train_model.py")
# MAGIC 
# MAGIC # Run the training script
# MAGIC %python
# MAGIC import subprocess
# MAGIC result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
# MAGIC print("STDOUT:", result.stdout)
# MAGIC print("STDERR:", result.stderr)
# MAGIC print("Return code:", result.returncode)