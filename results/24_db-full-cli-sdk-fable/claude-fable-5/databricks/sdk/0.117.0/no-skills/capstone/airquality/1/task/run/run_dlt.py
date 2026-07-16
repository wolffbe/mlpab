import io
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import pipelines, workspace

w = WorkspaceClient()
me = w.current_user.me().user_name
CAT, SCH = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]

nb_path = f"/Users/{me}/{PREFIX}/airq_dlt"
with open("run/airq_dlt.py", "rb") as fh:
    w.workspace.upload(nb_path, io.BytesIO(fh.read()),
                       format=workspace.ImportFormat.SOURCE,
                       language=workspace.Language.PYTHON, overwrite=True)
print("notebook uploaded:", nb_path, flush=True)

created = w.pipelines.create(
    name=f"{PREFIX}_airq_dlt",
    catalog=CAT,
    schema=SCH,
    serverless=True,
    development=True,
    continuous=False,
    libraries=[pipelines.PipelineLibrary(
        notebook=pipelines.NotebookLibrary(path=nb_path))],
    environment=pipelines.PipelinesEnvironment(
        dependencies=["scikit-learn", "mlflow"]),
)
pid = created.pipeline_id
print("pipeline created:", pid, flush=True)

upd = w.pipelines.start_update(pipeline_id=pid, full_refresh=True)
print("update started:", upd.update_id, flush=True)

while True:
    u = w.pipelines.get_update(pipeline_id=pid, update_id=upd.update_id).update
    print("state:", u.state, flush=True)
    if str(u.state) in ("UpdateInfoState.COMPLETED", "UpdateInfoState.FAILED",
                        "UpdateInfoState.CANCELED"):
        break
    time.sleep(20)

if str(u.state) != "UpdateInfoState.COMPLETED":
    print("--- events ---", flush=True)
    for ev in w.pipelines.list_pipeline_events(pipeline_id=pid, max_results=50):
        if ev.level and str(ev.level) in ("EventLevel.ERROR", "EventLevel.WARN"):
            print(ev.timestamp, ev.level, ev.message, flush=True)
            if ev.error:
                for ex in ev.error.exceptions or []:
                    print("   ", ex.message, flush=True)
    raise SystemExit("pipeline update failed")
print("PIPELINE COMPLETED", flush=True)
