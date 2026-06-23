#!/usr/bin/env python3
import hopsworks

# Login
hopsworks.login()

# Get project
project = hopsworks.project
print(f"Project: {project}")
print(f"Project type: {type(project)}")

# List all attributes of project
attrs = [x for x in dir(project) if not x.startswith('_')]
print(f"Project attributes: {attrs}")

# Try to get feature store from project
if hasattr(project, 'get_feature_store'):
    fs = project.get_feature_store()
    print(f"Got feature store via project.get_feature_store: {type(fs)}")
elif hasattr(project, 'feature_store'):
    fs = project.feature_store
    print(f"Got feature store via project.feature_store: {type(fs)}")
else:
    print("No feature store access found in project")
    raise Exception("Cannot access feature store")

print("Success!")
