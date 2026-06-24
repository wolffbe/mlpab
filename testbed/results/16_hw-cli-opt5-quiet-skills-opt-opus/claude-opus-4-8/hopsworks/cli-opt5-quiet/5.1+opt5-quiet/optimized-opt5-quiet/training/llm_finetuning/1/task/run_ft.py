"""Job entrypoint: stage inputs into CWD, run the provided fine-tune script
unchanged, push outputs back to HopsFS. Runs on the Hopsworks platform."""
import os
import runpy

import hopsworks

BASE = "Resources/ftjobf21faf"
OUT = BASE + "/out"

project = hopsworks.login()
ds = project.get_dataset_api()

print("CWD:", os.getcwd())

# Stage inputs (+ the unmodified script) into the working directory.
for f in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    if os.path.exists(f):
        os.remove(f)
    ds.download(BASE + "/" + f, local_path=f, overwrite=True)
    print("downloaded", f, os.path.getsize(f))

# Run the provided script byte-for-byte, as its own __main__.
runpy.run_path("finetune_model.py", run_name="__main__")

print("metrics.json contents:")
with open("metrics.json") as fh:
    print(fh.read())

# Push outputs back to HopsFS so they survive the executor.
try:
    ds.mkdir(OUT)
except Exception as e:
    print("mkdir:", e)
for f in ["finetuned_model.npz", "metrics.json"]:
    ds.upload(f, OUT, overwrite=True)
    print("uploaded", f)

print("DONE")
