"""Job driver: stages inputs into the job working dir, runs the provided
fine-tuning script AS-IS, and pushes the produced artifacts back to HopsFS.

This file does no ML work itself: it only moves files and invokes
finetune_model.py unchanged (via runpy, honoring its __main__ guard).
"""
import json
import os
import runpy

import hopsworks

REMOTE_DIR = "Resources/ftjob8025df"
OUT_DIR = "Resources/ftjob8025df/out"
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]

project = hopsworks.login()
ds = project.get_dataset_api()

# Stage inputs into the current working directory.
for name in INPUTS:
    if os.path.exists(name):
        os.remove(name)
    ds.download(f"{REMOTE_DIR}/{name}", local_path=name, overwrite=True)
    print(f"staged {name}: {os.path.getsize(name)} bytes", flush=True)

# Run the provided script exactly as-is.
runpy.run_path("finetune_model.py", run_name="__main__")

# Report and persist the outputs.
with open("metrics.json") as fh:
    metrics = json.load(fh)
print("METRICS_JSON_BEGIN", flush=True)
print(json.dumps(metrics), flush=True)
print("METRICS_JSON_END", flush=True)

try:
    ds.mkdir(OUT_DIR)
except Exception as exc:  # already exists is fine
    print(f"mkdir note: {exc}", flush=True)

for name in ["finetuned_model.npz", "metrics.json"]:
    ds.upload(name, OUT_DIR, overwrite=True)
    print(f"uploaded {name} -> {OUT_DIR}", flush=True)

print("DRIVER_DONE", flush=True)
