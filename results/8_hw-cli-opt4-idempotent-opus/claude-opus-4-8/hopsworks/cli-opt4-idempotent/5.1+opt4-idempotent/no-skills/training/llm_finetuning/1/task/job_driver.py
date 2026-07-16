"""Job driver: stage inputs, run the provided fine-tune script as-is, publish outputs.

Runs on the Hopsworks platform as job `ftjobc710b4`. It downloads the staged
input files into the working directory, executes the unmodified
finetune_model.py via runpy (run_name="__main__"), then uploads the produced
finetuned_model.npz and metrics.json back to HopsFS.
"""
import os
import runpy

import hopsworks

STAGE = "Resources/ftjobc710b4"
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]


def main():
    project = hopsworks.login()
    ds = project.get_dataset_api()

    cwd = os.getcwd()
    print("cwd:", cwd)

    for name in INPUTS:
        remote = f"{STAGE}/{name}"
        local = os.path.join(cwd, name)
        if os.path.exists(local):
            os.remove(local)
        ds.download(remote, local_path=local, overwrite=True)
        print("downloaded", remote, "->", local, os.path.getsize(local), "bytes")

    # Run the provided script exactly as-is.
    runpy.run_path(os.path.join(cwd, "finetune_model.py"), run_name="__main__")

    for name in OUTPUTS:
        local = os.path.join(cwd, name)
        print("produced", name, os.path.getsize(local), "bytes")
        ds.upload(local, STAGE, overwrite=True)
        print("uploaded", name, "->", f"{STAGE}/{name}")

    with open(os.path.join(cwd, "metrics.json")) as fh:
        print("metrics.json:", fh.read())


if __name__ == "__main__":
    main()
