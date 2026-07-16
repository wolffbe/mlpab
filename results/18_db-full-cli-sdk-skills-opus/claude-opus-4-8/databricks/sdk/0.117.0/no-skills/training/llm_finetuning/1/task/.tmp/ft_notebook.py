# Databricks notebook source
# Fine-tune job: runs the provided deterministic finetune script AS-IS on the platform.
import os, shutil, runpy, tempfile

STAGE = "/Volumes/workspace/mlpab5b087a/ftstage"
WORK = tempfile.mkdtemp(prefix="ftwork_")
for f in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    shutil.copy(os.path.join(STAGE, f), os.path.join(WORK, f))

os.chdir(WORK)
# Run the provided script unmodified, as if executed as a program.
runpy.run_path(os.path.join(WORK, "finetune_model.py"), run_name="__main__")

# Copy outputs back to the volume so the driver can read them.
for f in ["finetuned_model.npz", "metrics.json"]:
    shutil.copy(os.path.join(WORK, f), os.path.join(STAGE, f))

print("DONE")
print(open(os.path.join(WORK, "metrics.json")).read())
