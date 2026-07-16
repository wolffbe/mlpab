import databricks.sdk as dsdk
w = dsdk.WorkspaceClient()
base = "/Volumes/workspace/mlpab2138eb/trainjoba834e5_data"
for e in w.files.list_directory_contents(base):
    print(e.path, e.file_size)
resp = w.files.download(f"{base}/predictions.csv")
data = resp.contents.read().decode()
lines = data.splitlines()
print("rows:", len(lines))
print("\n".join(lines[:5]))
