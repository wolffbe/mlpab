import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("predictions646af0", version=1)
print("name:", fg.name, "| version:", fg.version)
print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("features:", [(f.name, f.type) for f in fg.features])
js = project.get_job_api()
job = js.get_job("trainjob646af0")
print("job:", job.name, "| last execution:", js.last_execution(job).state)
