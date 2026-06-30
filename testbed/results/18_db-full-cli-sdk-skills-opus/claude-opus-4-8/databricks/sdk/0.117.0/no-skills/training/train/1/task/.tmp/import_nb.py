import base64
import databricks.sdk as dsdk
from databricks.sdk.service import workspace as ws

w = dsdk.WorkspaceClient()
user = w.current_user.me().user_name
folder = f"/Users/{user}/mlpab2138eb"
w.workspace.mkdirs(folder)
nb_path = f"{folder}/trainjoba834e5_runner"

with open(".tmp/run_notebook_src.py", "rb") as f:
    content = f.read()

w.workspace.import_(
    path=nb_path,
    format=ws.ImportFormat.SOURCE,
    language=ws.Language.PYTHON,
    content=base64.b64encode(content).decode(),
    overwrite=True,
)
print("imported notebook:", nb_path)
o = w.workspace.get_status(nb_path)
print(o.object_type, o.path, o.language)
