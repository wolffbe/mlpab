"""Job wrapper: stages inputs into a writable workdir, runs the provided
finetune script as-is (unmodified), copies outputs back to the dataset dir.
Runs ON the Hopsworks platform as job ftjobc00779.
"""
import os
import shutil
import tempfile
import runpy

# Locate the dataset dir that holds this wrapper + inputs (fuse-mounted).
HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]
OUTPUTS = ["finetuned_model.npz", "metrics.json"]

work = tempfile.mkdtemp(prefix="ftjob_")
for f in INPUTS:
    shutil.copy(os.path.join(HERE, f), os.path.join(work, f))
    print("staged", f, os.path.getsize(os.path.join(work, f)))

os.chdir(work)
print("workdir", os.getcwd())

# Run the provided script exactly as-is (its __name__ == '__main__' guard fires).
runpy.run_path(os.path.join(work, "finetune_model.py"), run_name="__main__")

for f in OUTPUTS:
    src = os.path.join(work, f)
    print("output", f, os.path.getsize(src))
    shutil.copy(src, os.path.join(HERE, f))
    print("copied back", f)

with open(os.path.join(work, "metrics.json")) as fh:
    print("METRICS:", fh.read())
print("WRAPPER_DONE")
