import os
import sys

# Change to the directory where the data files are
os.chdir("/Workspace/Users/benedict@logicalclocks.com/mlpab64367b/")

# Now run the fine-tuning script
import finetune_model

if __name__ == "__main__":
    finetune_model.main()
