# Databricks notebook source
# MAGIC %pip install databricks-sdk --upgrade -q
dbutils.library.restartPython()

# COMMAND ----------
import inspect
import databricks.sdk.service.database as db_svc
from databricks.sdk import WorkspaceClient

results = []
w = WorkspaceClient()

# Check NewPipelineSpec structure
try:
    np_cls = db_svc.NewPipelineSpec
    sig = inspect.signature(np_cls.__init__)
    params = list(sig.parameters.keys())
    results.append(f"NewPipelineSpec params: {params}")
    # Try to instantiate
    try:
        inst = np_cls()
        results.append(f"NewPipelineSpec dict: {inst.as_dict()}")
    except Exception as e:
        results.append(f"NewPipelineSpec instantiation: {e}")
except AttributeError:
    results.append("NewPipelineSpec: not found in database module")
    # Try pipelines module
    import databricks.sdk.service.pipelines as pl_svc
    np_attrs = [a for a in dir(pl_svc) if 'new' in a.lower() or 'Pipeline' in a]
    results.append(f"pipelines module attrs: {np_attrs[:20]}")

spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.new_pipeline_spec_output")
