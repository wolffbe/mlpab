import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
catalog, sch = schema.split(".")

try:
    w.schemas.get(schema)
    print("schema exists:", schema)
except Exception:
    w.schemas.create(name=sch, catalog_name=catalog)
    print("schema created:", schema)

vol_full = f"{schema}.prediction_logs"
try:
    w.volumes.read(vol_full)
    print("volume exists:", vol_full)
except Exception:
    w.volumes.create(catalog_name=catalog, schema_name=sch,
                     name="prediction_logs", volume_type=VolumeType.MANAGED)
    print("volume created:", vol_full)

path = f"/Volumes/{catalog}/{sch}/prediction_logs/prediction_log.csv"
with open("data/prediction_log.csv", "rb") as f:
    w.files.upload(path, f, overwrite=True)
print("uploaded:", path)
