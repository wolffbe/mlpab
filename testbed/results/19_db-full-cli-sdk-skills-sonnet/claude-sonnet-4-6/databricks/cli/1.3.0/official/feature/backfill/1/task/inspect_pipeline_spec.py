# Databricks notebook source
# MAGIC %pip install databricks-sdk --upgrade -q
dbutils.library.restartPython()

# COMMAND ----------
import inspect
import databricks.sdk.service.database as db_svc
import databricks.sdk.service.pipelines as pl_svc

results = []

# Check PipelineSpec fields
for cls_name in ['PipelineSpec', 'PipelineLibrary', 'PipelineCluster']:
    try:
        cls = getattr(pl_svc, cls_name)
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.keys())
        results.append(f"{cls_name}: {params}")
    except AttributeError:
        results.append(f"{cls_name}: not found")

# Check if there's a serverless option in pipelines
pipeline_attrs = [a for a in dir(pl_svc) if 'server' in a.lower() or 'storage' in a.lower()]
results.append(f"serverless/storage attrs: {pipeline_attrs}")

# Check what new_pipeline_spec is in SyncedTableSpec
sig = inspect.signature(db_svc.SyncedTableSpec.__init__)
for name, param in sig.parameters.items():
    results.append(f"SyncedTableSpec.{name}: annotation={param.annotation}")

spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.pipeline_spec_output")
