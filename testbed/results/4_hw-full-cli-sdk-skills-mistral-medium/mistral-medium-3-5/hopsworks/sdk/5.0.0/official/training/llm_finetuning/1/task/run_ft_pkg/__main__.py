import os
import sys

# Change to the Resources directory where the data files are
os.chdir('/Resources')

# Run the fine-tuning script using os.system
os.system('python finetune_model.py')
