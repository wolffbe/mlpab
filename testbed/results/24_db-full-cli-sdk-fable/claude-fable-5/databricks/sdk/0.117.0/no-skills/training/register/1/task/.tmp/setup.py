"""Local driver: create volume + workspace dir, upload artifact/metrics/job script."""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType

w = WorkspaceClient()
me = w.current_user.me().user_name

try:
    w.volumes.create(
        catalog_name="workspace",
        schema_name="mlpab3d22c1",
        name="artifacts",
        volume_type=VolumeType.MANAGED,
    )
    print("volume created")
except Exception as e:
    print("volume create:", e)

w.workspace.mkdirs(f"/Users/{me}/mlpab3d22c1")
print("workspace dir ok")

for local, remote in [
    ("data/model.json", "model.json"),
    ("data/metrics.json", "metrics.json"),
    (".tmp/register_job.py", "register_job.py"),
]:
    with open(local, "rb") as fh:
        w.files.upload(
            f"/Volumes/workspace/mlpab3d22c1/artifacts/{remote}", fh, overwrite=True
        )
    print("uploaded", remote)

print([f.path for f in w.files.list_directory_contents("/Volumes/workspace/mlpab3d22c1/artifacts")])
