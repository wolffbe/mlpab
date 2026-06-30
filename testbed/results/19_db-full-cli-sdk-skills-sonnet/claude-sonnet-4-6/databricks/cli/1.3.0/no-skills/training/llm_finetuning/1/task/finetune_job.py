"""Fine-tuning job wrapper — copies data from volume, runs finetune_model.py, saves results back."""
import os
import shutil
import subprocess
import json

VOLUME_PATH = "/Volumes/workspace/mlpabeb2ad5/mlpabeb2ad5_finetune"
WORK_DIR = "/tmp/ftjob0b3133"

os.makedirs(WORK_DIR, exist_ok=True)

for fname in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    shutil.copy(f"{VOLUME_PATH}/{fname}", f"{WORK_DIR}/{fname}")

result = subprocess.run(
    ["python", "finetune_model.py"],
    cwd=WORK_DIR,
    capture_output=True,
    text=True
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
if result.returncode != 0:
    raise RuntimeError(f"finetune_model.py failed with exit code {result.returncode}")

shutil.copy(f"{WORK_DIR}/finetuned_model.npz", f"{VOLUME_PATH}/finetuned_model.npz")
shutil.copy(f"{WORK_DIR}/metrics.json", f"{VOLUME_PATH}/metrics.json")

with open(f"{WORK_DIR}/metrics.json") as f:
    metrics = json.load(f)

print("Metrics:", metrics)
