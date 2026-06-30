import inspect
import databricks.sdk.service.catalog as cat
for n in ["OnlineTable", "OnlineTableSpec", "OnlineTableSpecTriggeredSchedulingPolicy",
          "OnlineTableSpecContinuousSchedulingPolicy"]:
    obj = getattr(cat, n, None)
    print("==", n, "==")
    if obj is None:
        print("  MISSING")
        continue
    try:
        print(inspect.signature(obj.__init__))
    except Exception as e:
        print("  sig err", e)
import databricks.sdk as dsdk
print("create sig:", inspect.signature(dsdk.WorkspaceClient().online_tables.create))
