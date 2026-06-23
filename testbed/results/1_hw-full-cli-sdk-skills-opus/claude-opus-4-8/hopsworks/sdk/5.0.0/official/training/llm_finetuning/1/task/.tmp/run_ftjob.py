"""Job entry point: runs the provided finetune script UNMODIFIED, then
persists its outputs (finetuned_model.npz, metrics.json) to HopsFS so they
can be retrieved after the job's working directory is gone."""
import os
import hopsworks

# base_model.npz, finetune.txt, eval.txt and finetune_model.py are staged into
# the working directory via the job's `files` config. Run the script as-is.
import finetune_model
finetune_model.main()

assert os.path.exists("finetuned_model.npz"), "finetune did not produce model"
assert os.path.exists("metrics.json"), "finetune did not produce metrics"

proj = hopsworks.login()
ds = proj.get_dataset_api()
out_dir = "/Projects/%s/Resources/ftjobc00779_out" % proj.name
try:
    ds.mkdir(out_dir)
except Exception as e:
    print("mkdir note:", e)
ds.upload("finetuned_model.npz", out_dir, overwrite=True)
ds.upload("metrics.json", out_dir, overwrite=True)
print("UPLOADED outputs to", out_dir)
print(open("metrics.json").read())
