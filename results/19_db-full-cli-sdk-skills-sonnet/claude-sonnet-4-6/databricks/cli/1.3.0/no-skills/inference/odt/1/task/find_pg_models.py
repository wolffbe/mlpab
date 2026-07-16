# Databricks notebook source
# COMMAND ----------
import json
import inspect

results = {}

# Check databricks.sdk for postgres/lakebase related modules
try:
    import databricks.sdk.service.catalog as cat
    import databricks.sdk.service as svc
    import databricks.sdk

    # Look for lakebase, postgres, synced table models
    all_svc = [m for m in dir(svc) if not m.startswith('_')]
    results["sdk_services"] = all_svc

    # Check if there's a postgres-specific service module
    if hasattr(svc, 'postgres'):
        pg_svc = svc.postgres
        results["pg_svc_attrs"] = dir(pg_svc)

    # Find all databricks.sdk modules
    import pkgutil
    import databricks.sdk.service
    modules = [m.name for m in pkgutil.iter_modules(databricks.sdk.service.__path__)]
    results["sdk_service_modules"] = modules

except Exception as e:
    results["import_error"] = str(e)

# Check the main SDK service for postgres models
try:
    from databricks.sdk.service import postgres as pg_mod
    pg_classes = [x for x in dir(pg_mod) if not x.startswith('_')]
    results["pg_module_classes"] = pg_classes

    # Find the SyncedTable class
    if hasattr(pg_mod, 'SyncedTable'):
        st = pg_mod.SyncedTable
        results["SyncedTable_fields"] = list(st.__dataclass_fields__.keys()) if hasattr(st, '__dataclass_fields__') else str(inspect.signature(st.__init__))

    # Find other relevant classes
    for cls_name in pg_classes[:20]:
        cls = getattr(pg_mod, cls_name)
        if hasattr(cls, '__dataclass_fields__'):
            results[f"{cls_name}_fields"] = list(cls.__dataclass_fields__.keys())

except Exception as e:
    results["pg_module_error"] = str(e)

dbutils.notebook.exit(json.dumps(results))
