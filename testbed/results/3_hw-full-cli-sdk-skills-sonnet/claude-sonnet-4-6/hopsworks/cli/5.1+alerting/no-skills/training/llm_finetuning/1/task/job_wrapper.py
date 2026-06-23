"""Wrapper for fine-tuning job — runs script from HopsFS mount directory."""
import os
import subprocess
import sys

# The HopsFS mount has all input files; run the script from there
os.chdir("/hopsfs/Resources/jobs/ftjob30461d")

result = subprocess.run([sys.executable, "finetune_model.py"], check=True)

print("Fine-tuning complete. Outputs written to HopsFS directory.")
