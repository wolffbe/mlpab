"""Vertex CustomJob wrapper.

Downloads the fine-tuning inputs and the UNMODIFIED finetune_model.py from GCS
into the working directory, executes finetune_model.py exactly as-is (as
__main__), then uploads the produced finetuned_model.npz and metrics.json back
to GCS. The provided script is run verbatim via runpy — not edited.
"""
import os
import runpy

from google.cloud import storage


def _split(gs):
    assert gs.startswith("gs://")
    bucket, _, prefix = gs[5:].partition("/")
    return bucket, prefix.rstrip("/")


def main():
    in_uri = os.environ["JOB_INPUT_URI"]
    out_uri = os.environ["JOB_OUTPUT_URI"]
    client = storage.Client()

    in_bucket_name, in_prefix = _split(in_uri)
    in_bucket = client.bucket(in_bucket_name)
    for fname in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
        blob = in_bucket.blob(f"{in_prefix}/{fname}")
        blob.download_to_filename(fname)
        print(f"downloaded {fname}")

    # Run the provided script exactly as-is, as __main__.
    runpy.run_path("finetune_model.py", run_name="__main__")

    out_bucket_name, out_prefix = _split(out_uri)
    out_bucket = client.bucket(out_bucket_name)
    for fname in ["finetuned_model.npz", "metrics.json"]:
        blob = out_bucket.blob(f"{out_prefix}/{fname}")
        blob.upload_from_filename(fname)
        print(f"uploaded {fname}")
    print("JOB_DONE")


if __name__ == "__main__":
    main()
