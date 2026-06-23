import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import hopsworks
from hsfs.feature import Feature

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]

project = hopsworks.login()
fs = project.get_feature_store()

# --- read history ---
hist = pd.read_csv("data/airquality_history.csv")
hist["event_time"] = pd.to_datetime(hist["date"])
hist["date"] = hist["date"].astype(str)
for c in FEATURES + ["pm25"]:
    hist[c] = hist[c].astype("float64")
print("history rows:", len(hist), "cols:", list(hist.columns))

fg = fs.get_or_create_feature_group(
    name="airq963ee7",
    version=1,
    description="Daily air-quality + weather features with measured pm25 target",
    primary_key=["date"],
    event_time="event_time",
    online_enabled=True,
    stream=True,
    features=[
        Feature("date", "string", description="Calendar day (record key)"),
        Feature("event_time", "timestamp", description="Day the values were valid"),
        Feature("pm25_lag1", "double", description="Previous day's measured pm25"),
        Feature("temperature", "double", description="Daily mean temperature (C)"),
        Feature("humidity", "double", description="Relative humidity (%)"),
        Feature("wind_speed", "double", description="Wind speed"),
        Feature("pressure", "double", description="Atmospheric pressure (hPa)"),
        Feature("precipitation", "double", description="Precipitation (mm)"),
        Feature("pm25", "double", description="Measured PM2.5 (target)"),
    ],
)
fg.insert(hist, wait=True)
print("inserted history into airq963ee7; fg.id=", fg.id)

# --- feature view + training dataset airqtd963ee7 ---
query = fg.select(FEATURES + ["pm25"])
try:
    fv = fs.get_feature_view(name="airqtd963ee7", version=1)
    print("FV already exists")
except Exception:
    fv = fs.create_feature_view(
        name="airqtd963ee7",
        version=1,
        description="Training view: weather + lag pm25 signals -> pm25",
        query=query,
        labels=["pm25"],
    )
    print("created FV airqtd963ee7")

td_version, job = fv.create_train_test_split(
    test_size=0.2, seed=42, statistics_config=True,
    description="airq pm25 train/test split",
    write_options={"wait_for_job": True},
)
print("TRAINING_DATASET_VERSION:", td_version)
