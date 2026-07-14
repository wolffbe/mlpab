import hw_env  # noqa: F401
import hopsworks

proj = hopsworks.login()
ds = proj.get_dataset_api()

print("Resources/submission/answers.json exists:", ds.exists("Resources/submission/answers.json"))
try:
    print("content:", ds.read_content("Resources/submission/answers.json").content[:200])
except Exception as e:
    print("read_content failed:", e)

# Also try a top-level submission path for good measure.
try:
    ds.mkdir("submission")
except Exception as e:
    print("mkdir submission:", str(e)[:150])
try:
    ds.upload("submission/answers.json", "submission", overwrite=True)
    print("top-level upload ok:", ds.exists("submission/answers.json"))
except Exception as e:
    print("top-level upload failed:", str(e)[:200])
