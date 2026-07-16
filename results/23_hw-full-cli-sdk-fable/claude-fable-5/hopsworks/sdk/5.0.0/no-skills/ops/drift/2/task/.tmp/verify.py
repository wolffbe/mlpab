import hw_env  # noqa: F401
import hopsworks

proj = hopsworks.login()
ds = proj.get_dataset_api()
for p in ("submission/answers.json", "Resources/submission/answers.json"):
    try:
        local = ds.download(p, ".tmp/verify_dl", overwrite=True)
        print(p, "->", open(local).read())
    except Exception as e:
        print(p, "download failed:", str(e)[:150])
