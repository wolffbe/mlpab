#!/usr/bin/env python3
"""Wrapper script to copy data files and run training script."""
import os
import shutil
import subprocess
import sys

# Copy data files to current working directory
workspace_path = "/Users/benedict@logicalclocks.com/mlpab155832/data"
current_dir = os.getcwd()

print(f"Current working directory: {current_dir}")
print(f"Copying files from: {workspace_path}")

# Copy the data files
files_to_copy = ["train.csv", "score.csv", "train_model.py"]
for filename in files_to_copy:
    src = os.path.join(workspace_path, filename)
    dst = os.path.join(current_dir, filename)
    print(f"Copying {src} to {dst}")
    shutil.copy2(src, dst)

# Run the training script
print("Running training script...")
result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)

# Copy predictions back to workspace
predictions_src = os.path.join(current_dir, "predictions.csv")
predictions_dst = os.path.join(workspace_path, "predictions.csv")
if os.path.exists(predictions_src):
    print(f"Copying predictions.csv to {predictions_dst}")
    shutil.copy2(predictions_src, predictions_dst)
else:
    print("predictions.csv not found!")
    sys.exit(1)