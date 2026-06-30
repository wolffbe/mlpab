# Databricks notebook source
import inspect
import databricks.sdk.service.database as db_svc

results = []

# Inspect key classes
for cls_name in ['SyncedDatabaseTable', 'SyncedTableSpec', 'DeltaTableSyncInfo', 'DatabaseAPI', 'SyncedTableSchedulingPolicy']:
    try:
        cls = getattr(db_svc, cls_name)
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.keys())
        results.append(f"{cls_name} params: {params}")
        # Try to instantiate to get dict
        try:
            inst = cls()
            results.append(f"{cls_name} dict: {inst.as_dict()}")
        except Exception as e:
            results.append(f"{cls_name} instantiation error: {e}")
    except AttributeError:
        results.append(f"{cls_name}: not found")

spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.class_inspection")
