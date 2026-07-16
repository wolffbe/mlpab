from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute

w = WorkspaceClient()
try:
    nts = w.clusters.list_node_types()
    names = [n.node_type_id for n in nts.node_types][:15]
    print("node types sample:", names)
except Exception as e:
    print("node types err:", e)

try:
    c = w.clusters.create(
        cluster_name="mlpab67db84_cc",
        spark_version="15.4.x-cpu-ml-scala2.12",
        num_workers=0,
        node_type_id="m5d.large",
        custom_tags={"ResourceClass": "SingleNode"},
        spark_conf={
            "spark.databricks.cluster.profile": "singleNode",
            "spark.master": "local[*]",
        },
        autotermination_minutes=30,
    )
    print("cluster creating:", c.cluster_id)
except Exception as e:
    print("cluster create err:", type(e).__name__, e)
