"""Vertex CustomJob wrapper: stage inputs from GCS, run the provided
finetune_model.py UNMODIFIED, upload outputs back to GCS."""
import os
import runpy
from google.cloud import storage

INPUT_URI = os.environ["INPUT_URI"].rstrip("/")
OUTPUT_URI = os.environ["OUTPUT_URI"].rstrip("/")


def split(uri):
    b, _, p = uri[len("gs://"):].partition("/")
    return b, p


client = storage.Client()

in_bucket, in_prefix = split(INPUT_URI)
bkt = client.bucket(in_bucket)
for fn in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    bkt.blob(f"{in_prefix}/{fn}").download_to_filename(fn)
    print("downloaded", fn, flush=True)

# Run the provided script exactly as-is from the working directory.
runpy.run_path("finetune_model.py", run_name="__main__")
print("finetune complete", flush=True)

out_bucket, out_prefix = split(OUTPUT_URI)
obkt = client.bucket(out_bucket)
for fn in ["finetuned_model.npz", "metrics.json"]:
    obkt.blob(f"{out_prefix}/{fn}").upload_from_filename(fn)
    print("uploaded", fn, flush=True)
print("DONE", flush=True)
