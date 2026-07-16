import hopsworks

project = hopsworks.login()
print("project:", project.name)
fs = project.get_feature_store()
print("fs:", fs.name)
print([m for m in dir(fs) if "sql" in m or "spine" in m or "create" in m])
