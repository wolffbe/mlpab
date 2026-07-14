import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
js = project.get_job_api()

wrapper = '''import os
import runpy

os.chdir("/hopsfs/Resources/trainjob646af0")
runpy.run_path("train_model.py", run_name="__main__")
print("wrote:", os.path.abspath("predictions.csv"), os.path.exists("predictions.csv"))
'''
with open("job_wrapper.py", "w") as fh:
    fh.write(wrapper)
print("uploaded:", ds.upload("job_wrapper.py", "Resources/trainjob646af0", overwrite=True))

job = js.get_job("trainjob646af0")
execution = job.run(await_termination=True)
print("final state:", execution.state, "| success:", execution.success)
print("predictions exists:", ds.exists("Resources/trainjob646af0/predictions.csv"))
