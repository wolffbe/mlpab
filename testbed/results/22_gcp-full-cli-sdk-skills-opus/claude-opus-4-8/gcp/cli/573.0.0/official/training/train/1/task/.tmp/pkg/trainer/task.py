"""Wrapper that runs the provided train_model.py unmodified as a Vertex job.

Downloads train.csv/score.csv from GCS into the script's directory, executes
train_model.py exactly as-is via subprocess (reads the CSVs from its working
directory and writes predictions.csv), then uploads predictions.csv to GCS.
"""
import os
import subprocess
import sys

from google.cloud import storage


def _split(gs_uri):
    assert gs_uri.startswith("gs://"), gs_uri
    bucket, blob = gs_uri[len("gs://"):].split("/", 1)
    return bucket, blob


def main():
    train_uri, score_uri, out_uri = sys.argv[1], sys.argv[2], sys.argv[3]
    here = os.path.dirname(os.path.abspath(__file__))
    client = storage.Client()

    b, blob = _split(train_uri)
    client.bucket(b).blob(blob).download_to_filename(os.path.join(here, "train.csv"))
    b, blob = _split(score_uri)
    client.bucket(b).blob(blob).download_to_filename(os.path.join(here, "score.csv"))

    # Run the provided script exactly as-is, from the dir holding the CSVs.
    subprocess.run([sys.executable, "train_model.py"], check=True, cwd=here)

    b, blob = _split(out_uri)
    client.bucket(b).blob(blob).upload_from_filename(os.path.join(here, "predictions.csv"))
    print("PREDICTIONS_UPLOADED", out_uri)


if __name__ == "__main__":
    main()
