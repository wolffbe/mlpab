import databricks.sdk as dsdk
w = dsdk.WorkspaceClient()
me = w.current_user.me()
print("user:", me.user_name)
attrs = [a for a in dir(w) if not a.startswith('_')]
print(attrs)
