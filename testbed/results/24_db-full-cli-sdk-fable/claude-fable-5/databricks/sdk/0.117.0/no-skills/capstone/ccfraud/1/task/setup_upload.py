import io
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType

w = WorkspaceClient()
CAT, SCH = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")

# make sure schema exists
try:
    s = w.schemas.get(f"{CAT}.{SCH}")
    print("schema ok:", s.full_name)
except Exception as e:
    print("schema get failed:", e)
    w.schemas.create(name=SCH, catalog_name=CAT)
    print("schema created")

# volume for raw data
try:
    w.volumes.create(catalog_name=CAT, schema_name=SCH, name="raw", volume_type=VolumeType.MANAGED)
    print("volume created")
except Exception as e:
    print("volume create:", e)

for fname in ("transactions.csv", "score_transactions.csv"):
    with open(f"data/{fname}", "rb") as f:
        w.files.upload(f"/Volumes/{CAT}/{SCH}/raw/{fname}", f, overwrite=True)
    print("uploaded", fname)

for fi in w.files.list_directory_contents(f"/Volumes/{CAT}/{SCH}/raw"):
    print(fi.path, fi.file_size)
