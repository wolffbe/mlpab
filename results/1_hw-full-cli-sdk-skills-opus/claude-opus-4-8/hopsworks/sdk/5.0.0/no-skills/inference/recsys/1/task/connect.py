import hopsworks
proj = hopsworks.login()
print("PROJECT:", proj.name)
fs = proj.get_feature_store()
print("FS type:", type(fs))
print("FS methods:", [m for m in dir(fs) if not m.startswith('_')])
