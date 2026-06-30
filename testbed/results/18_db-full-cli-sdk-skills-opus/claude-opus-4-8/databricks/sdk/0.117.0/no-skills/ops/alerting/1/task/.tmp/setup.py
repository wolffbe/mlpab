import io
import databricks.sdk as dbsdk
from databricks.sdk.service.workspace import ImportFormat

w = dbsdk.WorkspaceClient()
user = w.current_user.me().user_name
prefix = "mlpab34dc41"
base_dir = "/Users/" + user + "/" + prefix
w.workspace.mkdirs(base_dir)

src = open("data/failing_job.py", "rb").read()
remote_path = base_dir + "/flaky62cc43.py"
w.workspace.upload(remote_path, io.BytesIO(src), format=ImportFormat.AUTO, overwrite=True)
print("uploaded ->", remote_path)
print("exists:", w.workspace.get_status(remote_path).object_type)
