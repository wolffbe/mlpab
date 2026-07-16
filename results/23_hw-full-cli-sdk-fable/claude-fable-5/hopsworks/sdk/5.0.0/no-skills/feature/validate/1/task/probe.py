import hopsworks

print(hopsworks.__version__)
project = hopsworks.login()
print("project:", project.name)
fs = project.get_feature_store()
print("feature store:", fs.name)
