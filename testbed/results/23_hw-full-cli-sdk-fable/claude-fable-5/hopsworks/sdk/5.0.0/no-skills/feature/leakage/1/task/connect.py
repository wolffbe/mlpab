import hopsworks

project = hopsworks.login()
print("project:", project.name)
fs = project.get_feature_store()
print("feature store:", fs.name)
