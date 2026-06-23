"""Job wrapper: download inputs, run the provided fine-tune script as-is, upload outputs.

The provided data/finetune_model.py is NOT modified. It is executed verbatim via
runpy with run_name="__main__" so its own `if __name__ == "__main__"` guard fires
and main() runs with the original frozen hyperparameters. Inputs are pulled from
HopsFS into the job's CWD (the working dir the script reads from); outputs
(finetuned_model.npz, metrics.json) are pushed back to HopsFS.
"""
import os
import runpy

import hopsworks

HOPSFS_DIR = "Resources/ftjobcda62f"
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]


def main():
    project = hopsworks.login()
    ds = project.get_dataset_api()

    workdir = os.getcwd()
    print(f"Job working directory: {workdir}", flush=True)

    for name in INPUTS:
        dst = os.path.join(workdir, name)
        if os.path.exists(dst):
            os.remove(dst)
        ds.download(f"{HOPSFS_DIR}/{name}", local_path=dst, overwrite=True)
        print(f"Downloaded {name} ({os.path.getsize(dst)} bytes)", flush=True)

    # Run the provided script verbatim, as __main__, from the CWD it reads/writes.
    runpy.run_path(os.path.join(workdir, "finetune_model.py"), run_name="__main__")

    for name in OUTPUTS:
        src = os.path.join(workdir, name)
        print(f"Produced {name} ({os.path.getsize(src)} bytes)", flush=True)
        ds.upload(src, HOPSFS_DIR, overwrite=True)
        print(f"Uploaded {name} to {HOPSFS_DIR}", flush=True)

    with open(os.path.join(workdir, "metrics.json")) as fh:
        print("metrics.json contents:", fh.read(), flush=True)


if __name__ == "__main__":
    main()
