import warnings
warnings.filterwarnings("ignore")
import hopsworks

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("airq963ee7", version=1)

fv = fs.get_feature_view(name="airqtd963ee7", version=1)
if fv is None:
    fv = fs.create_feature_view(
        name="airqtd963ee7",
        version=1,
        description="Training view: weather + lag pm25 signals -> pm25",
        query=fg.select(FEATURES + ["pm25"]),
        labels=["pm25"],
    )
    print("created FV airqtd963ee7")
else:
    print("FV exists:", fv.name, fv.version)

td_version, job = fv.create_train_test_split(
    test_size=0.2, seed=42, statistics_config=True,
    description="airq pm25 train/test split",
    write_options={"wait_for_job": True},
)
print("TRAINING_DATASET_VERSION:", td_version)
