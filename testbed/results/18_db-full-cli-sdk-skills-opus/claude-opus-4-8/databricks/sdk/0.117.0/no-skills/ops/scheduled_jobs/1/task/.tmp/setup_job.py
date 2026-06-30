import base64
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

w = WorkspaceClient()
me = w.current_user.me().user_name
base = f"/Users/{me}/mlpab8d3f26"
w.workspace.mkdirs(base)
nb_path = f"{base}/heartbeat468788"

code = (
    "# Databricks notebook source\n"
    "import datetime\n"
    'TOKEN = "HB-56927584"\n'
    'msg = "heartbeat " + TOKEN + " alive at " + datetime.datetime.now(datetime.timezone.utc).isoformat()\n'
    "import sys; sys.stdout.write(msg + chr(10))\n"
)

w.workspace.import_(
    path=nb_path,
    content=base64.b64encode(code.encode()).decode(),
    language=workspace.Language.PYTHON,
    format=workspace.ImportFormat.SOURCE,
    overwrite=True,
)
st = w.workspace.get_status(nb_path)
print("Uploaded notebook:", nb_path, "type:", st.object_type)
