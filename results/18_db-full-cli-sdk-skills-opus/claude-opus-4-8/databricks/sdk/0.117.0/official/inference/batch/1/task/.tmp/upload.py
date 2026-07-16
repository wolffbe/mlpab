import databricks.sdk as s
import io, os

w = s.WorkspaceClient()
schema_full = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab1ab216
catalog, schema = schema_full.split(".")
vol = "ingest"

# create volume
from databricks.sdk.service.catalog import VolumeType
try:
    w.volumes.create(catalog_name=catalog, schema_name=schema, name=vol,
                     volume_type=VolumeType.MANAGED)
    print("volume created")
except Exception as e:
    print("volume create:", repr(e)[:200])

vol_path = f"/Volumes/{catalog}/{schema}/{vol}/feature_history.csv"
with open("data/feature_history.csv", "rb") as f:
    data = f.read()
w.files.upload(vol_path, io.BytesIO(data), overwrite=True)
print("uploaded to", vol_path, "bytes", len(data))
