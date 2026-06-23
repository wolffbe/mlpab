"""Setup and run the training job on Hopsworks platform."""
import os
import time
import hopsworks

JOB_NAME = "trainjob7a8727"

def main():
    print("Connecting to Hopsworks...")
    project = hopsworks.login()
    print(f"Connected to project: {project.name}")

    dataset_api = project.get_dataset_api()

    # Upload script and data files to HopsFS
    print("Uploading files to HopsFS...")

    # Create directory in Resources
    upload_dir = f"Resources/{JOB_NAME}"
    try:
        dataset_api.mkdir(upload_dir)
    except Exception as e:
        print(f"mkdir warning (may already exist): {e}")

    # Upload each file to the job's resource directory
    for fname in ["train_model.py", "train.csv", "score.csv"]:
        local_path = f"data/{fname}" if fname != "train_model.py" else f"data/{fname}"
        hops_path = f"{upload_dir}/{fname}"
        print(f"  Uploading {fname}...")
        dataset_api.upload(local_path, upload_dir, overwrite=True)

    # Create job via SDK
    print("Creating job...")
    job_api = project.get_job_api()

    config = job_api.get_configuration("PYTHON")
    print(f"Default PYTHON config keys: {list(config.keys())}")

    config["appPath"] = f"/Projects/{project.name}/{upload_dir}/train_model.py"
    config["environmentName"] = "pandas-training-pipeline"

    # Try to set additional files so CSVs are in working dir
    # In some Hopsworks versions this can be set as files or resourceFiles
    print(f"Job config: {config}")

    job = job_api.create_job(JOB_NAME, config)
    print(f"Job created: {job.name}")

    # Run job and wait for termination
    print("Starting job execution...")
    execution = job.run(await_termination=True)
    print(f"Execution state: {execution.state}")
    print(f"Execution final status: {execution.final_status}")

    return execution


if __name__ == "__main__":
    main()
