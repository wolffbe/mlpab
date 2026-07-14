#!/usr/bin/env python
import os
import sys

# Change to the directory containing the data files
# In Databricks workspace, the path is /Users/... not /Workspace/Users/...
work_dir = "/Users/benedict@logicalclocks.com/mlpab64367b/"
os.chdir(work_dir)

# Read and execute the finetune_model.py script directly
with open(os.path.join(work_dir, "finetune_model.py"), "r") as f:
    script_code = f.read()

# Execute the script in the current namespace
exec(script_code)

# Call main if it exists
if "main" in globals():
    main()
