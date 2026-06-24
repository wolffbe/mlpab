"""Job wrapper: fetch inputs into CWD, run the provided fine-tune script
unchanged, then upload its outputs back to HopsFS. Runs on the platform."""
import os
import runpy

import hopsworks

REMOTE_DIR = "Resources/ftjob4d1d84"
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]


def main():
    project = hopsworks.login()
    ds = project.get_dataset_api()
    cwd = os.getcwd()
    print("Working directory:", cwd, flush=True)

    for name in INPUTS:
        local = os.path.join(cwd, name)
        if os.path.exists(local):
            os.remove(local)
        ds.download(f"{REMOTE_DIR}/{name}", local, overwrite=True)
        print("Downloaded", name, os.path.getsize(local), "bytes", flush=True)

    # Run the provided fine-tuning script EXACTLY as-is (as __main__).
    runpy.run_path(os.path.join(cwd, "finetune_model.py"), run_name="__main__")

    for name in OUTPUTS:
        local = os.path.join(cwd, name)
        print("Produced", name, os.path.getsize(local), "bytes", flush=True)
        ds.upload(local, REMOTE_DIR, overwrite=True)
        print("Uploaded", name, flush=True)

    with open(os.path.join(cwd, "metrics.json")) as f:
        print("metrics.json contents:", f.read(), flush=True)


if __name__ == "__main__":
    main()
