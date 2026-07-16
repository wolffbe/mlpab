import hopsworks
project = hopsworks.login()
api = project.get_trino_api()
print("TRINO API TYPE:", type(api))
for m in dir(api):
    if not m.startswith("__"):
        print("  ", m)
