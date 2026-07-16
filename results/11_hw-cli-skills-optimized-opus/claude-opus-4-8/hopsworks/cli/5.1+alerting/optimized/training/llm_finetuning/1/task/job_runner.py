"""Platform job wrapper: fetch inputs, run the provided fine-tune script
AS-IS via runpy, then store the outputs back to HopsFS. The provided
finetune_model.py is NOT modified — runpy executes its __main__ block."""
import os
import hopsworks

BASE = "Resources/ftjobb59376"
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]


def main():
    project = hopsworks.login()
    ds = project.get_dataset_api()
    cwd = os.getcwd()
    print("CWD:", cwd)

    for f in INPUTS:
        dst = os.path.join(cwd, f)
        if os.path.exists(dst):
            os.remove(dst)
        ds.download(f"{BASE}/{f}", local_path=cwd, overwrite=True)
        print("downloaded", f, "exists=", os.path.exists(dst))

    import runpy
    runpy.run_path(os.path.join(cwd, "finetune_model.py"), run_name="__main__")

    for f in OUTPUTS:
        path = os.path.join(cwd, f)
        print("output", f, "exists=", os.path.exists(path), "size=",
              os.path.getsize(path) if os.path.exists(path) else -1)
        ds.upload(path, BASE, overwrite=True)
        print("uploaded", f)

    with open(os.path.join(cwd, "metrics.json")) as fh:
        print("METRICS:", fh.read())
    print("JOB_DONE")


if __name__ == "__main__":
    main()
