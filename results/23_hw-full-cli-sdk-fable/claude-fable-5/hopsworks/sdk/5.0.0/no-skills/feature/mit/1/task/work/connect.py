import hopsworks

project = hopsworks.login()
print("project:", project.name)
print("version:", hopsworks.__version__)
print([a for a in dir(project) if not a.startswith("_")])
