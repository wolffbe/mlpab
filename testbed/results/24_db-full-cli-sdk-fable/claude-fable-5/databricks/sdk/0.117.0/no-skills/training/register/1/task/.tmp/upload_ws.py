"""Upload register_job.py as a workspace file."""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat

w = WorkspaceClient()
me = w.current_user.me().user_name
path = f"/Users/{me}/mlpab3d22c1/register_job.py"
with open(".tmp/register_job.py", "rb") as fh:
    w.workspace.upload(path, fh, format=ImportFormat.AUTO, overwrite=True)
print("uploaded to", path)
print(w.workspace.get_status(path))
