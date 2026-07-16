from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
user = w.current_user.me().user_name
folder = f"/Users/{user}/mlpab23fe6e"
w.workspace.mkdirs(folder)

src = open("data/heartbeat.py", "rb").read()
path = f"{folder}/heartbeat"
w.workspace.upload(path, src, format=ImportFormat.SOURCE, language=Language.PYTHON, overwrite=True)
print("uploaded:", path)
print(w.workspace.get_status(path))
