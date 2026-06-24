"""Job entrypoint for ftjob9845cc.

Runs ENTIRELY on the Hopsworks platform. Downloads the fine-tuning inputs from
HopsFS into the job's working directory, runs the PROVIDED finetune_model.py
unmodified (via its main()), then uploads the produced finetuned_model.npz and
metrics.json back to HopsFS so they can be retrieved and registered.
"""
import os
import sys

import hopsworks

HOPSFS_DIR = "Resources/ftjob9845cc"
INPUTS = ["finetune_model.py", "base_model.npz", "finetune.txt", "eval.txt"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]


def main():
    project = hopsworks.login()
    ds = project.get_dataset_api()

    cwd = os.getcwd()
    print("Job working directory:", cwd)

    for fn in INPUTS:
        dst = os.path.join(cwd, fn)
        if os.path.exists(dst):
            os.remove(dst)
        ds.download(f"{HOPSFS_DIR}/{fn}", dst, overwrite=True)
        print("downloaded", fn, os.path.getsize(dst), "bytes")

    # Run the provided script as-is, in this working directory.
    sys.path.insert(0, cwd)
    import finetune_model
    finetune_model.main()

    for fn in OUTPUTS:
        src = os.path.join(cwd, fn)
        print("produced", fn, os.path.getsize(src), "bytes")
        ds.upload(src, HOPSFS_DIR, overwrite=True)
        print("uploaded", fn)

    with open(os.path.join(cwd, "metrics.json")) as f:
        print("metrics.json contents:", f.read())
    print("FTJOB_DONE")


if __name__ == "__main__":
    main()
