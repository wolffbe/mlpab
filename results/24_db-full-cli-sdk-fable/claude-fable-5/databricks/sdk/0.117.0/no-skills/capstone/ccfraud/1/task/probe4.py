from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

cfg = Config(http_timeout_seconds=30)
w = WorkspaceClient(config=cfg)
print("clusters:")
for c in w.clusters.list():
    print(" ", c.cluster_id, c.cluster_name, c.state)
print("done")
