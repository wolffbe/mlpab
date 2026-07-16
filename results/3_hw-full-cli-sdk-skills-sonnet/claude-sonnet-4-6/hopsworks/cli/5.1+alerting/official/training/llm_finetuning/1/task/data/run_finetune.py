"""Wrapper script to run finetune_model.py as a Hopsworks job."""
import os
import subprocess
import sys

work_dir = os.getcwd()
print(f"Working directory: {work_dir}", flush=True)
print(f"Files present: {os.listdir(work_dir)}", flush=True)

print("Running finetune_model.py ...", flush=True)
result = subprocess.run([sys.executable, "finetune_model.py"])
if result.returncode != 0:
    raise RuntimeError(f"finetune_model.py exited with code {result.returncode}")

print("Fine-tuning complete.", flush=True)
print(f"Output files: {[f for f in os.listdir(work_dir) if f in ('finetuned_model.npz', 'metrics.json')]}", flush=True)
