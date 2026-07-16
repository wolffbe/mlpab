"""Platform job driver for ftjobc00779.

Stages the fine-tune inputs into the job's working directory, runs the
PROVIDED, UNMODIFIED finetune_model.py (imported and executed as-is), then
uploads finetuned_model.npz and metrics.json back to HopsFS so they can be
read back through the platform.
"""
import os
import runpy

import hopsworks

PROJECT_BASE = "Resources/ftjobc00779"
INPUTS = PROJECT_BASE + "/inputs"
OUTPUTS = PROJECT_BASE + "/outputs"
INPUT_FILES = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUT_FILES = ["finetuned_model.npz", "metrics.json"]

project = hopsworks.login()
ds = project.get_dataset_api()

workdir = os.getcwd()
print("Working directory:", workdir)

# Stage inputs into the working directory next to the script.
for name in INPUT_FILES:
    remote = "{}/{}".format(INPUTS, name)
    local = os.path.join(workdir, name)
    if os.path.exists(local):
        os.remove(local)
    ds.download(remote, local)
    print("downloaded", remote, "->", local)

# Run the provided fine-tuning script exactly as-is, as __main__.
runpy.run_path(os.path.join(workdir, "finetune_model.py"), run_name="__main__")
print("fine-tune complete; outputs:", os.listdir(workdir))

# Persist outputs back to HopsFS.
for name in OUTPUT_FILES:
    local = os.path.join(workdir, name)
    ds.upload(local, OUTPUTS, overwrite=True)
    print("uploaded", local, "->", OUTPUTS)

print("done")
