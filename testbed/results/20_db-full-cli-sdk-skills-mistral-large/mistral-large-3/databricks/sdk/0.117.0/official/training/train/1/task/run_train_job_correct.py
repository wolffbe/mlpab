"""Script to create and run a Databricks job using the SDK correctly."""
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")
JOB_NAME = "trainjobac536a"

# Initialize the WorkspaceClient
w = WorkspaceClient()

# Define the job using JobCluster and ClusterSpec
job = w.jobs.create(
    name=JOB_NAME,
    job_clusters=[
        jobs.JobCluster(
            job_cluster_key="train_cluster",
            new_cluster=jobs.ClusterSpec(
                spark_version="14.3.x-scala2.12",
                node_type_id="i3.xlarge",
                num_workers=0,
                spark_conf={
                    "spark.databricks.cluster.profile": "singleNode",
                    "spark.master": "local[*]",
                },
                custom_tags={"resource_class": "single-node"},
            ),
        )
    ],
    tasks=[
        jobs.Task(
            task_key="train_task",
            job_cluster_key="train_cluster",
            spark_python_task=jobs.SparkPythonTask(
                python_file="dbfs:/FileStore/train_model.py",
            ),
            libraries=[
                compute.Library(pypi=compute.PythonPyPiLibrary(package="pandas==2.0.3")),
                compute.Library(pypi=compute.PythonPyPiLibrary(package="numpy==1.24.3")),
            ],
            timeout_seconds=3600,
        )
    ],
    max_concurrent_runs=1,
)

print(f"Created job with ID: {job.job_id}")

# Upload the training script and data to DBFS
w.dbfs.mkdirs("dbfs:/FileStore/")
w.dbfs.upload("./data/train_model.py", "dbfs:/FileStore/train_model.py", overwrite=True)
w.dbfs.upload("./data/train.csv", "dbfs:/FileStore/train.csv", overwrite=True)
w.dbfs.upload("./data/score.csv", "dbfs:/FileStore/score.csv", overwrite=True)

# Start the job
run = w.jobs.run_now(job_id=job.job_id).result()
print(f"Started job run with ID: {run.run_id}")

# Wait for the job to complete
while True:
    run_status = w.jobs.get_run(run_id=run.run_id)
    if run_status.state.life_cycle_state == jobs.RunLifeCycleState.TERMINATED:
        if run_status.state.result_state == jobs.RunResultState.SUCCESS:
            print("Job completed successfully.")
            break
        else:
            raise Exception(f"Job failed with state: {run_status.state.result_state}")
    time.sleep(10)

# Download the predictions file
w.dbfs.download(f"dbfs:/FileStore/predictions.csv", "./predictions.csv", overwrite=True)
print("Downloaded predictions.csv")