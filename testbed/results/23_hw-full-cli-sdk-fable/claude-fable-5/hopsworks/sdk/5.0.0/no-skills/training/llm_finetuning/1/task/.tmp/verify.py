import hopsworks

project = hopsworks.login()
mr = project.get_model_registry()
m = mr.get_model("ftmodel6f5e78", version=1)
print("model:", m.name, "version:", m.version)
print("metrics:", m.training_metrics)
print("path:", m.model_files_path if hasattr(m, "model_files_path") else m.version_path)
ds = project.get_dataset_api()
files_path = m.model_files_path if hasattr(m, "model_files_path") else None
if files_path:
    rel = files_path.split("/Projects/" + project.name + "/", 1)[-1]
    print("files:", [f.path for f in ds.list(rel)["items"]] if isinstance(ds.list(rel), dict) else ds.list(rel))
job = project.get_job_api().get_job("ftjob6f5e78")
ex = job.get_executions()[0]
print("job:", job.name, "last exec:", ex.final_status)
