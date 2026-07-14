import hopsworks

project = hopsworks.login()
print("project:", project.name)
fs = project.get_feature_store()
print("fs:", fs.name)
print([m for m in dir(fs) if not m.startswith("_")])
