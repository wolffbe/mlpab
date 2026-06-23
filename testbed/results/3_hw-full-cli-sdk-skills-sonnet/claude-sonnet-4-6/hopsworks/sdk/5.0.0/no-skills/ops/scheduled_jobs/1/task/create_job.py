"""Create a recurring scheduled heartbeat job on Hopsworks."""
import hopsworks
import json
import os
import time

project = hopsworks.login()
jobs_api = project.get_jobs_api()

print("Available in jobs_api:", dir(jobs_api))
help(jobs_api)
