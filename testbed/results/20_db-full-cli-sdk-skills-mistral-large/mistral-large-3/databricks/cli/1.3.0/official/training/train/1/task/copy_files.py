"""Copy train.csv and score.csv to the working directory."""
import shutil
import os

# Source paths in the workspace
prefix = "mlpaba53a68"
src_train = "/Workspace/Users/benedict@logicalclocks.com/{}/train.csv".format(prefix)
src_score = "/Workspace/Users/benedict@logicalclocks.com/{}/score.csv".format(prefix)

# Destination paths in the working directory
dst_train = "train.csv"
dst_score = "score.csv"

# Copy files
shutil.copy(src_train, dst_train)
shutil.copy(src_score, dst_score)

print("Copied train.csv and score.csv to working directory.")