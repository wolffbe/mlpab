import io, os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language
from databricks.sdk.service.catalog import VolumeType

w = WorkspaceClient()
catalog, schema = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]

# volume
try:
    w.volumes.create(catalog_name=catalog, schema_name=schema, name="taskvol", volume_type=VolumeType.MANAGED)
    print("volume created")
except Exception as e:
    print("volume:", e)

vol = f"/Volumes/{catalog}/{schema}/taskvol"
for fn in ["train.csv", "score.csv", "train_model.py"]:
    with open(f"data/{fn}", "rb") as f:
        w.files.upload(f"{vol}/{fn}", f, overwrite=True)
    print("uploaded", fn)

# wrapper notebook: runs the provided script UNMODIFIED with cwd containing the CSVs
nb = f'''# Databricks notebook source
import os, shutil, runpy
work = "/tmp/trainwork178367"
os.makedirs(work, exist_ok=True)
vol = "{vol}"
shutil.copy(f"{{vol}}/train.csv", work)
shutil.copy(f"{{vol}}/score.csv", work)
os.chdir(work)
runpy.run_path(f"{{vol}}/train_model.py", run_name="__main__")
shutil.copy("predictions.csv", f"{{vol}}/predictions.csv")
print("done, predictions written to", vol)
'''
nb_dir = f"/Users/{w.current_user.me().user_name}/{prefix}"
w.workspace.mkdirs(nb_dir)
nb_path = f"{nb_dir}/run_train178367"
w.workspace.upload(nb_path, io.BytesIO(nb.encode()), format=ImportFormat.SOURCE, language=Language.PYTHON, overwrite=True)
print("notebook:", nb_path)
