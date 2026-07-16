import databricks.sdk

w = databricks.sdk.WorkspaceClient()
attrs = [a for a in dir(w) if not a.startswith("_")]
print(attrs)
print()
print("me:", w.current_user.me().user_name)
