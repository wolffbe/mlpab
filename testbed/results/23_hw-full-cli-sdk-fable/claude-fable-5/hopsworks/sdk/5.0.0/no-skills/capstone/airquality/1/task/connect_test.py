import hopsworks

print("hopsworks version:", hopsworks.__version__)
project = hopsworks.login()
print("project:", project.name)
print(sorted(m for m in dir(project) if not m.startswith("_")))
