"""Job wrapper: localize inputs, run the provided finetune script AS-IS, push outputs.

Runs on the Hopsworks platform as job `ftjob24d8fe`. It does not modify the
fine-tuning logic in any way: it downloads the required inputs into the working
directory and then executes data/finetune_model.py verbatim via runpy with
run_name="__main__", which is exactly equivalent to `python finetune_model.py`.
"""
import os
import runpy

import hopsworks

HOPS_DIR = "Resources/ftjob24d8fe"
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]

project = hopsworks.login()
ds = project.get_dataset_api()

workdir = os.getcwd()
print("Working directory:", workdir, flush=True)

for name in INPUTS:
    if os.path.exists(name):
        os.remove(name)
    local = ds.download(f"{HOPS_DIR}/{name}", local_path=os.path.join(workdir, name), overwrite=True)
    print("Downloaded", name, "->", local, flush=True)

# Run the provided script exactly as-is (unmodified, hyperparameters untouched).
runpy.run_path(os.path.join(workdir, "finetune_model.py"), run_name="__main__")

for name in OUTPUTS:
    path = os.path.join(workdir, name)
    assert os.path.exists(path), f"expected output missing: {name}"
    ds.upload(path, HOPS_DIR, overwrite=True)
    print("Uploaded output", name, flush=True)

with open(os.path.join(workdir, "metrics.json")) as fh:
    print("metrics.json:", fh.read(), flush=True)
print("DONE", flush=True)
