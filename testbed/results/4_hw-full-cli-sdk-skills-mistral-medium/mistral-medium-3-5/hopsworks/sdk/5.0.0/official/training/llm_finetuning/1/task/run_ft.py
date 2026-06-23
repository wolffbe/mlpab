import os
import sys
import importlib.util

# Change to the Resources directory where the data files are
os.chdir('/Resources')

# Load and run the finetune_model module
spec = importlib.util.spec_from_file_location("finetune_model", "finetune_model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main()
