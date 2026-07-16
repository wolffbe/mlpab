import io
import databricks.sdk as dsdk
from databricks.sdk.service import catalog

w = dsdk.WorkspaceClient()
CATALOG = "workspace"
SCHEMA = "mlpab2138eb"
VOL = "trainjoba834e5_data"

# Create volume (managed) if not exists
try:
    v = w.volumes.create(catalog_name=CATALOG, schema_name=SCHEMA, name=VOL,
                         volume_type=catalog.VolumeType.MANAGED)
    print("created volume", v.full_name)
except Exception as e:
    print("volume create:", repr(e)[:300])

base = f"/Volumes/{CATALOG}/{SCHEMA}/{VOL}"
for fn in ["train.csv", "score.csv", "train_model.py"]:
    with open(f"data/{fn}", "rb") as f:
        data = f.read()
    w.files.upload(f"{base}/{fn}", io.BytesIO(data), overwrite=True)
    print("uploaded", fn, len(data))

print("LISTING:")
for e in w.files.list_directory_contents(base):
    print(" ", e.path, e.file_size)
