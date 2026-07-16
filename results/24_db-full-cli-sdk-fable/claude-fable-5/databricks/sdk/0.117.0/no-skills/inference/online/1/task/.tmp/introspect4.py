import inspect

import databricks.sdk.service.database as db
import databricks.sdk.service.ml as ml
import databricks.sdk.service.catalog as cat

for cls in [db.DatabaseInstance, db.SyncedDatabaseTable, db.SyncedTableSpec,
            db.NewPipelineSpec, db.DatabaseCatalog]:
    print("=" * 70)
    print(cls.__name__)
    sig = inspect.signature(cls.__init__)
    for p in sig.parameters.values():
        if p.name != "self":
            print("  ", p)

print("=" * 70)
print("SyncedTableSchedulingPolicy:", [m.name for m in db.SyncedTableSchedulingPolicy])
print("OnlineStore fields:")
for p in inspect.signature(ml.OnlineStore.__init__).parameters.values():
    if p.name != "self":
        print("  ", p)
print("PublishSpec fields:")
for p in inspect.signature(ml.PublishSpec.__init__).parameters.values():
    if p.name != "self":
        print("  ", p)
print("OnlineTable fields:")
for p in inspect.signature(cat.OnlineTable.__init__).parameters.values():
    if p.name != "self":
        print("  ", p)
print("OnlineTableSpec fields:")
for p in inspect.signature(cat.OnlineTableSpec.__init__).parameters.values():
    if p.name != "self":
        print("  ", p)
