import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
js = project.get_job_api()

base = "Resources/trainjob646af0"
if not ds.exists(base):
    ds.mkdir(base)

for f in ["train.csv", "score.csv", "train_model.py"]:
    path = ds.upload("data/" + f, base, overwrite=True)
    print("uploaded:", path)

# wrapper: fetch inputs, run provided script unmodified, persist output
wrapper = '''import runpy
import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
base = "Resources/trainjob646af0"
for f in ["train.csv", "score.csv", "train_model.py"]:
    ds.download(base + "/" + f, f, overwrite=True)
runpy.run_path("train_model.py", run_name="__main__")
ds.upload("predictions.csv", base, overwrite=True)
print("done")
'''
with open("job_wrapper.py", "w") as fh:
    fh.write(wrapper)
print("uploaded:", ds.upload("job_wrapper.py", base, overwrite=True))

cfg = js.get_configuration("PYTHON")
print("PYTHON job config template:", cfg)
