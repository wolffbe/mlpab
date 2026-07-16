"""Platform job wrapper for the provided fine-tuning script.

Runs ON the Hopsworks platform as job `ftjobde791b`. It localises the three
input files plus the unchanged `finetune_model.py` into the job's working
directory, executes that script AS-IS (so its hyperparameters and logic are
untouched), then uploads the produced artifacts back to HopsFS so they can be
retrieved and registered.
"""
import os
import runpy

import hopsworks

BASE = "Resources/ftjobde791b"
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]


def main():
    project = hopsworks.login()
    ds = project.get_dataset_api()

    cwd = os.getcwd()
    print("Working directory:", cwd)

    for f in INPUTS:
        ds.download(f"{BASE}/{f}", local_path=os.path.join(cwd, f), overwrite=True)
        print("Downloaded", f)

    # Run the provided script unchanged, triggering its __main__ block.
    runpy.run_path(os.path.join(cwd, "finetune_model.py"), run_name="__main__")

    for f in OUTPUTS:
        assert os.path.exists(f), f"expected output missing: {f}"
        ds.upload(os.path.join(cwd, f), BASE, overwrite=True)
        print("Uploaded", f)

    with open("metrics.json") as fh:
        print("metrics.json:", fh.read())


if __name__ == "__main__":
    main()
