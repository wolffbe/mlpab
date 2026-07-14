import json
import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute, jobs

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
cat, sch = schema.split(".")
vol = f"/Volumes/{cat}/{sch}/drift_vol"

# schema + volume via UC APIs (no warehouse needed)
try:
    w.schemas.create(name=sch, catalog_name=cat)
except Exception as e:
    print("schema:", e)
try:
    from databricks.sdk.service.catalog import VolumeType
    w.volumes.create(catalog_name=cat, schema_name=sch, name="drift_vol",
                     volume_type=VolumeType.MANAGED)
except Exception as e:
    print("volume:", e)

with open("data/features.csv", "rb") as f:
    w.files.upload(f"{vol}/features.csv", f, overwrite=True)
with open("drift_job.py", "rb") as f:
    w.files.upload(f"{vol}/drift_job.py", f, overwrite=True)
print("uploaded")

run = w.jobs.submit_and_wait(
    run_name=f"{prefix}_drift_analysis",
    tasks=[
        jobs.SubmitTask(
            task_key="drift",
            environment_key="default",
            spark_python_task=jobs.SparkPythonTask(
                python_file=f"{vol}/drift_job.py", parameters=[vol]
            ),
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="default",
            spec=compute.Environment(environment_version="3"),
        )
    ],
    timeout=None,
)
print("run state:", run.state)

resp = w.files.download(f"{vol}/answers.json")
ans = json.loads(resp.contents.read())
print("ANSWER:", ans)
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump(ans, f)

resp = w.files.download(f"{vol}/daily_means.json")
print(resp.contents.read().decode()[:3000])
