"""Job entrypoint: stage inputs, run the provided finetune script AS-IS, publish outputs.

Runs ON the Hopsworks platform. Downloads the three task inputs and the
unmodified finetune_model.py from HopsFS into the job working directory, then
executes finetune_model.py exactly as its author intended (run_name="__main__"
so its `if __name__ == "__main__"` guard fires and nothing is altered), then
uploads finetuned_model.npz and metrics.json back to HopsFS.
"""
import os
import runpy

import hopsworks

REMOTE_DIR = "Resources/ftjobcbc135"
OUTPUT_DIR = "Resources/ftjobcbc135/output"
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]

project = hopsworks.login()
ds = project.get_dataset_api()

cwd = os.getcwd()
print("Job working directory:", cwd, flush=True)

for name in INPUTS:
    remote = f"{REMOTE_DIR}/{name}"
    local = os.path.join(cwd, name)
    if os.path.exists(local):
        os.remove(local)
    ds.download(remote, local_path=local, overwrite=True)
    print("downloaded", remote, "->", local, os.path.getsize(local), "bytes", flush=True)

# Run the provided script unmodified, in this working directory.
runpy.run_path(os.path.join(cwd, "finetune_model.py"), run_name="__main__")

for name in OUTPUTS:
    local = os.path.join(cwd, name)
    print("produced", name, os.path.getsize(local), "bytes", flush=True)
    ds.upload(local, OUTPUT_DIR, overwrite=True)
    print("uploaded", name, "->", OUTPUT_DIR, flush=True)

with open(os.path.join(cwd, "metrics.json")) as f:
    print("metrics.json:", f.read(), flush=True)
print("JOB_WRAPPER_DONE", flush=True)
