# Databricks notebook source
# This is a Databricks notebook that will copy files and run the fine-tuning script

# Cell 1: Copy files from workspace to /tmp using dbutils
import os
import sys
import json
import base64

os.makedirs('/tmp', exist_ok=True)
sys.path.insert(0, '/tmp')

# Use dbutils to copy files
# In Databricks, workspace files can be accessed via file:/Workspace/...
dbutils.fs.cp('file:/Workspace/Users/benedict@logicalclocks.com/mlpab64367b/finetune_model.py', 'file:/tmp/finetune_model.py')
dbutils.fs.cp('file:/Workspace/Users/benedict@logicalclocks.com/mlpab64367b/base_model.npz', 'file:/tmp/base_model.npz')
dbutils.fs.cp('file:/Workspace/Users/benedict@logicalclocks.com/mlpab64367b/finetune.txt', 'file:/tmp/finetune.txt')
dbutils.fs.cp('file:/Workspace/Users/benedict@logicalclocks.com/mlpab64367b/eval.txt', 'file:/tmp/eval.txt')

print("Files copied to /tmp")

# Cell 2: Run the fine-tuning script
os.chdir('/tmp')
print(f"Current directory: {os.getcwd()}")

# Run the script
import finetune_model
finetune_model.main()

# Cell 3: Upload output files back to workspace
# Read metrics.json
with open('/tmp/metrics.json', 'r') as f:
    metrics = json.load(f)
    print(f"Metrics: {metrics}")

# Upload metrics.json to workspace
dbutils.fs.cp('file:/tmp/metrics.json', 'file:/Workspace/Users/benedict@logicalclocks.com/mlpab64367b/metrics.json')
dbutils.fs.cp('file:/tmp/finetuned_model.npz', 'file:/Workspace/Users/benedict@logicalclocks.com/mlpab64367b/finetuned_model.npz')

print("Done!")
