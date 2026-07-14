"""Wrapper that runs the provided training script unmodified on Vertex AI.

Downloads the task inputs (train.csv, score.csv, train_model.py) from GCS into
the working directory, executes train_model.py exactly as provided, then uploads
the resulting predictions.csv back to GCS.
"""
import os
import subprocess
import sys

from google.cloud import storage

BUCKET = os.environ["JOB_BUCKET"]
IN_PREFIX = os.environ["JOB_IN_PREFIX"]
OUT_URI = os.environ["JOB_OUT_URI"]  # gs://bucket/path/predictions.csv

client = storage.Client()
in_bucket = client.bucket(BUCKET)
for fn in ["train.csv", "score.csv", "train_model.py"]:
    in_bucket.blob(f"{IN_PREFIX}/{fn}").download_to_filename(fn)
    print(f"downloaded {fn}", flush=True)

# Run the provided, unmodified training script as-is.
subprocess.run([sys.executable, "train_model.py"], check=True)
print("training script finished", flush=True)

assert os.path.exists("predictions.csv"), "predictions.csv not produced"
out_bucket_name, out_path = OUT_URI[len("gs://"):].split("/", 1)
client.bucket(out_bucket_name).blob(out_path).upload_from_filename("predictions.csv")
print(f"uploaded predictions to {OUT_URI}", flush=True)
