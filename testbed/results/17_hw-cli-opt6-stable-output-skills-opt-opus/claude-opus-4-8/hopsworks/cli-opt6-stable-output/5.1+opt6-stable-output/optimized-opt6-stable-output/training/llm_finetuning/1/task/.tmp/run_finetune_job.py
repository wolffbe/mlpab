"""Job wrapper: stage inputs into CWD, run the provided finetune script
verbatim, then publish outputs back to HopsFS. The finetune logic itself is
NOT modified — we exec data/finetune_model.py as __main__ from the working
directory after placing its three required inputs there."""
import os
import runpy

import hopsworks

REMOTE_DIR = "Resources/ftjobd00657"
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]


def main():
    project = hopsworks.login()
    ds = project.get_dataset_api()

    workdir = os.getcwd()
    print("Working directory:", workdir, flush=True)

    for name in INPUTS:
        dst = os.path.join(workdir, name)
        if os.path.exists(dst):
            os.remove(dst)
        ds.download(f"{REMOTE_DIR}/{name}", local_path=workdir, overwrite=True)
        print("Downloaded", name, "->", dst, os.path.exists(dst), flush=True)

    # Run the provided script verbatim, as __main__, from this directory.
    runpy.run_path(os.path.join(workdir, "finetune_model.py"), run_name="__main__")

    for name in OUTPUTS:
        src = os.path.join(workdir, name)
        print("Output produced:", name, os.path.exists(src), flush=True)
        ds.upload(src, REMOTE_DIR, overwrite=True)
        print("Uploaded", name, "to", REMOTE_DIR, flush=True)

    with open(os.path.join(workdir, "metrics.json")) as f:
        print("metrics.json:", f.read(), flush=True)


if __name__ == "__main__":
    main()
