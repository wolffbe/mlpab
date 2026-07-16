#!/usr/bin/env python
import os
import sys

# Change to the directory containing the data files
work_dir = "/Workspace/Users/benedict@logicalclocks.com/mlpab64367b/"
os.chdir(work_dir)

# Add the directory to the path so we can import the module
sys.path.insert(0, work_dir)

# Import and run the main function
import finetune_model

if __name__ == "__main__":
    finetune_model.main()
