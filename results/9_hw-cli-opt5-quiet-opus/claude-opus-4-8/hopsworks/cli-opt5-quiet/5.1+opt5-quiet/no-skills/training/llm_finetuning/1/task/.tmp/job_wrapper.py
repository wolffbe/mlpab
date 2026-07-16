"""Platform job wrapper for ftjobf34327.

Runs the PROVIDED, UNMODIFIED data/finetune_model.py inside a Hopsworks job.
Steps (all on the platform):
  1. Connect to the project (in-cluster login).
  2. Download base_model.npz, finetune.txt, eval.txt and finetune_model.py
     from Resources/ftjobf34327/ into the job's working directory.
  3. Execute finetune_model.py as-is via runpy (its __main__ guard runs main()).
  4. Upload the produced finetuned_model.npz and metrics.json back to
     Resources/ftjobf34327/ so the driver can read them out.
"""
import os
import runpy

import hopsworks

REMOTE_DIR = "Resources/ftjobf34327"
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]


def main():
    project = hopsworks.login()
    ds = project.get_dataset_api()

    workdir = os.getcwd()
    print("Working directory:", workdir, flush=True)

    for name in INPUTS:
        local = os.path.join(workdir, name)
        if os.path.exists(local):
            os.remove(local)
        ds.download(f"{REMOTE_DIR}/{name}", local, overwrite=True)
        print("downloaded", name, os.path.getsize(local), "bytes", flush=True)

    # Run the provided fine-tuning script unmodified, in this working dir.
    runpy.run_path(os.path.join(workdir, "finetune_model.py"), run_name="__main__")

    for name in OUTPUTS:
        local = os.path.join(workdir, name)
        print("produced", name, os.path.getsize(local), "bytes", flush=True)
        ds.upload(local, REMOTE_DIR, overwrite=True)
        print("uploaded", name, flush=True)

    with open(os.path.join(workdir, "metrics.json")) as fh:
        print("metrics.json contents:", fh.read(), flush=True)


if __name__ == "__main__":
    main()
