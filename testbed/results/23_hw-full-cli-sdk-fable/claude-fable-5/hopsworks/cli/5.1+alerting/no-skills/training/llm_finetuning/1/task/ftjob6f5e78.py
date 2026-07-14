"""Driver for job ftjob6f5e78: stage inputs into the working directory,
run the provided finetune_model.py unmodified, upload the outputs back."""
import runpy

import hopsworks

REMOTE_DIR = "Resources/ftjob6f5e78"
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]


def main():
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()
    for name in INPUTS:
        dataset_api.download(f"{REMOTE_DIR}/{name}", overwrite=True)
    runpy.run_path("finetune_model.py", run_name="__main__")
    for name in OUTPUTS:
        dataset_api.upload(name, REMOTE_DIR, overwrite=True)


if __name__ == "__main__":
    main()
