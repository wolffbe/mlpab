import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
js = project.get_job_api()

base = "Resources/trainjob646af0"
print("uploaded:", ds.upload("data/train.csv", base, overwrite=True))
for p in ["train.csv", "score.csv", "train_model.py", "job_wrapper.py"]:
    ok = ds.exists(base + "/" + p)
    print(p, "->", ok)
    assert ok, p + " missing on hopsfs"

job = js.get_job("trainjob646af0")
execution = job.run(await_termination=True)
print("final state:", execution.state, "| success:", execution.success)
print("predictions exists:", ds.exists(base + "/predictions.csv"))
