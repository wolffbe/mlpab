import databricks.sdk.service.ml as ml
for n in ["OnlineStoreState", "PublishSpecPublishMode"]:
    obj = getattr(ml, n)
    print("==", n, "==")
    print([m for m in dir(obj) if not m.startswith('_')])
# capacity is a plain string; look in docstring of OnlineStore
print("OnlineStore doc:")
print((ml.OnlineStore.__doc__ or "")[:1500])
