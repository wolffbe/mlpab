#!/usr/bin/env python3
import os
import sys

# Change to the directory where the data files are
os.chdir('/Resources')

# Read and execute the finetune_model.py script
with open('finetune_model.py', 'r') as f:
    code = compile(f.read(), 'finetune_model.py', 'exec')
    exec(code)
