import hopsworks

BASE = "Resources/ftjob6f5e78"

project = hopsworks.login()
ds = project.get_dataset_api()

if not ds.path_exists(BASE):
    ds.mkdir(BASE)
for local in ["data/base_model.npz", "data/finetune.txt", "data/eval.txt",
              "data/finetune_model.py", ".tmp/ft_runner.py"]:
    p = ds.upload(local, BASE, overwrite=True)
    print("uploaded:", p)

job_api = project.get_job_api()
config = job_api.get_configuration("PYTHON")
config["appPath"] = "/" + BASE + "/ft_runner.py"
job = job_api.create_job("ftjob6f5e78", config)
print("job created:", job.name, job.job_type)
