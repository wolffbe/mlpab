import inspect
import databricks.sdk as dsdk
import databricks.sdk.service.ml as ml

w = dsdk.WorkspaceClient()

print("=== existing online stores ===")
try:
    for s in w.feature_store.list_online_stores():
        print(" ", s)
except Exception as e:
    print("  list err:", repr(e)[:300])

for n in ["OnlineStore", "PublishSpec", "OnlineStoreState", "PublishSpecPublishMode"]:
    obj = getattr(ml, n, None)
    print("==", n, "==")
    if obj is None:
        print("  MISSING from ml; trying other modules")
        continue
    try:
        print(" ", inspect.signature(obj.__init__))
    except Exception as e:
        print("  ", e)
