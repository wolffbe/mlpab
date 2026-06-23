import hopsworks, inspect
proj = hopsworks.login()
fs = proj.get_feature_store()
print("=== fs.sql sig ===")
print(inspect.signature(fs.sql))
print(fs.sql.__doc__)
print("=== create_feature_group sig ===")
print(inspect.signature(fs.create_feature_group))
