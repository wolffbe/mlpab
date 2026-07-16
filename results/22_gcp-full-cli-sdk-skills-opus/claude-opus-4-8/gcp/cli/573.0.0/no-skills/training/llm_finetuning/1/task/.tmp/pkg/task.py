"""Vertex custom-job entrypoint.

Stages inputs from GCS into the working directory, runs the provided
fine-tuning script AS-IS (imports its main()), then uploads the produced
finetuned_model.npz and metrics.json back to GCS.
"""
import argparse
import os

from google.cloud import storage

import finetune_model  # provided script, unmodified


def _split(gcs_uri):
    assert gcs_uri.startswith("gs://"), gcs_uri
    rest = gcs_uri[len("gs://"):]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs_input", required=True)
    ap.add_argument("--gcs_output", required=True)
    args = ap.parse_args()

    client = storage.Client()

    in_bucket, in_prefix = _split(args.gcs_input)
    bkt = client.bucket(in_bucket)
    for name in ("base_model.npz", "finetune.txt", "eval.txt"):
        blob = bkt.blob(f"{in_prefix}/{name}")
        blob.download_to_filename(name)
        print(f"downloaded gs://{in_bucket}/{in_prefix}/{name} ({os.path.getsize(name)} bytes)")

    # Run the provided fine-tuning script exactly as-is.
    finetune_model.main()

    out_bucket, out_prefix = _split(args.gcs_output)
    obkt = client.bucket(out_bucket)
    for name in ("finetuned_model.npz", "metrics.json"):
        blob = obkt.blob(f"{out_prefix}/{name}")
        blob.upload_from_filename(name)
        print(f"uploaded gs://{out_bucket}/{out_prefix}/{name}")

    with open("metrics.json") as fh:
        print("metrics.json:", fh.read())


if __name__ == "__main__":
    main()
