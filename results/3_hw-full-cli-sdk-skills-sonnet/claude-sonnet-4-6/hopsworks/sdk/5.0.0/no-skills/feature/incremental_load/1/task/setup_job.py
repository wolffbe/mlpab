import hopsworks
import json

project = hopsworks.login()

print("Connected to Hopsworks")

jobs_api = project.get_jobs_api()

# Get available job configurations
config = jobs_api.get_configuration("PYSPARK")
print("Default PYSPARK config:")
print(json.dumps(config, indent=2))
