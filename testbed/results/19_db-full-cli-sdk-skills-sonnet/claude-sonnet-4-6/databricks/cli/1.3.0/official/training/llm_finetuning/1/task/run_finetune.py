"""Job script: copies data from volume, runs fine-tuning, writes outputs back."""
import os
import shutil
import sys

VOLUME_PATH = "/Volumes/workspace/mlpab4886e6/mlpab4886e6_ft"
WORK_DIR = "/tmp/ft_workdir"

os.makedirs(WORK_DIR, exist_ok=True)

for fname in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    src = os.path.join(VOLUME_PATH, fname)
    dst = os.path.join(WORK_DIR, fname)
    shutil.copy2(src, dst)

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

import finetune_model
finetune_model.main()

for fname in ["finetuned_model.npz", "metrics.json"]:
    src = os.path.join(WORK_DIR, fname)
    dst = os.path.join(VOLUME_PATH, fname)
    shutil.copy2(src, dst)

print("Fine-tuning complete. Outputs written to volume.")
import json
with open("metrics.json") as f:
    print(json.load(f))
