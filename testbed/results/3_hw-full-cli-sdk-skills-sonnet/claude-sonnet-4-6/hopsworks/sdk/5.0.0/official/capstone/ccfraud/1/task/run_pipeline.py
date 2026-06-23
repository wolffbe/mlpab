#!/usr/bin/env python3
"""
Orchestration script: upload data to HopsFS, deploy and run the FTI pipeline job.
Runs locally using only the hopsworks SDK to drive the platform.
"""

import os
import hopsworks

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PIPELINE_SCRIPT = os.path.join(os.path.dirname(__file__), "pipeline.py")
HOPSFS_DIR = "Resources/ccfraud"
JOB_NAME = "ccfraud_pipeline"


def main():
    print("Connecting to Hopsworks …")
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()

    # ── Upload data files to HopsFS ───────────────────────────────────────────
    print(f"Uploading data to HopsFS: {HOPSFS_DIR}")
    try:
        dataset_api.mkdir(HOPSFS_DIR)
    except Exception as e:
        print(f"  mkdir: {e} (may already exist)")

    for fname in ["transactions.csv", "score_transactions.csv"]:
        local_path = os.path.join(DATA_DIR, fname)
        remote_path = HOPSFS_DIR
        print(f"  Uploading {fname} …")
        dataset_api.upload(local_path, remote_path, overwrite=True)
    print("Data uploaded.")

    # ── Upload pipeline script to HopsFS ──────────────────────────────────────
    print("Uploading pipeline.py …")
    dataset_api.upload(PIPELINE_SCRIPT, HOPSFS_DIR, overwrite=True)
    print("Script uploaded.")

    # ── Create and run job ────────────────────────────────────────────────────
    jobs_api = project.get_jobs_api()

    job_config = jobs_api.get_configuration("PYTHON")
    job_config["appPath"] = (
        f"/Projects/{project.name}/{HOPSFS_DIR}/pipeline.py"
    )
    # Use the pandas-training-pipeline env which has xgboost + sklearn
    job_config["defaultArgs"] = ""

    print(f"Creating/updating job '{JOB_NAME}' …")
    try:
        # Try to delete existing job first to allow clean re-create
        existing = jobs_api.get_job(JOB_NAME)
        print(f"Job exists, deleting to recreate …")
        existing.delete()
    except Exception:
        pass

    try:
        job = jobs_api.create_job(JOB_NAME, job_config)
        print("Job created.")
    except Exception as e:
        raise RuntimeError(f"Cannot create job: {e}") from e

    print(f"Running job '{JOB_NAME}' and waiting for completion …")
    execution = job.run(await_termination=True)

    print(f"\nJob finished. State: {execution.state}")
    success = str(execution.state).upper() in ("SUCCEEDED", "FINISHED", "SUCCESS")
    if not success:
        print(f"WARNING: job may have failed. Final state = {execution.state}")
    else:
        print("Pipeline job completed successfully!")

    return success


if __name__ == "__main__":
    ok = main()
    exit(0 if ok else 1)
