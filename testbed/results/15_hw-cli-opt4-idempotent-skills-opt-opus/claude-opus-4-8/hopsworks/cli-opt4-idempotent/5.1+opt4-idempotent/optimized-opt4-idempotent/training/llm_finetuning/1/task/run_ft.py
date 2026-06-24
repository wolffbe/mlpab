"""Job entrypoint: stage inputs, run the provided finetune script as-is, ship outputs.

This wrapper does NOT modify data/finetune_model.py or its hyperparameters. It
only downloads the inputs into a local working directory, invokes the script's
own main() (run as __main__ via runpy), and uploads the artifacts it produces
back to HopsFS.
"""
import os
import runpy
import shutil
import tempfile

import hopsworks

REMOTE_DIR = "Resources/ftjobaad365"
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]


def main():
    project = hopsworks.login()
    ds = project.get_dataset_api()

    workdir = tempfile.mkdtemp(prefix="ftjob_")
    print("Staging working directory:", workdir, flush=True)

    for name in INPUTS:
        remote = REMOTE_DIR + "/" + name
        dest = os.path.join(workdir, name)
        print("Downloading", remote, flush=True)
        path = ds.download(remote, local_path=dest, overwrite=True)
        print("  -> returned", path, "exists:", os.path.exists(dest), flush=True)

    print("workdir contents:", sorted(os.listdir(workdir)), flush=True)

    os.chdir(workdir)
    print("CWD now:", os.getcwd(), flush=True)

    # Run the provided script exactly as-is, as the program's __main__.
    runpy.run_path(os.path.join(workdir, "finetune_model.py"), run_name="__main__")

    print("post-run contents:", sorted(os.listdir(workdir)), flush=True)

    for name in OUTPUTS:
        local = os.path.join(workdir, name)
        if not os.path.exists(local):
            raise RuntimeError("Expected output missing: " + local)
        print("Uploading", local, "->", REMOTE_DIR, flush=True)
        ds.upload(local, REMOTE_DIR, overwrite=True)

    with open(os.path.join(workdir, "metrics.json")) as fh:
        print("metrics.json contents:", fh.read(), flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
