import databricks.sdk as s
w = s.WorkspaceClient()
attrs = [a for a in dir(w) if not a.startswith("_")]
print("WS attrs with 'data'/'sync'/'lake'/'database':")
for a in attrs:
    al = a.lower()
    if any(k in al for k in ("data","sync","lake","postgres","feature")):
        print("  ", a)
print("ALL:", attrs)
