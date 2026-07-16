import os
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
catalog, sch = schema.split(".")
path = f"/Volumes/{catalog}/{sch}/prediction_logs/answers.json"
with open("submission/answers.json", "rb") as f:
    w.files.upload(path, f, overwrite=True)
print("uploaded:", path)
