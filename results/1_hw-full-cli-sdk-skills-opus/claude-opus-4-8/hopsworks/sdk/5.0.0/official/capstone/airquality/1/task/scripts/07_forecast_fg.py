import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import hopsworks
from hsfs.feature import Feature

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]

project = hopsworks.login()
fs = project.get_feature_store()

ds = project.get_dataset_api()
try:
    print("listing Resources/airq:", ds.list("Resources/airq"))
except Exception as e:
    print("list err", repr(e))

fc = pd.read_csv("data/forecast_days.csv")
fc["event_time"] = pd.to_datetime(fc["date"])
fc["date"] = fc["date"].astype(str)
for c in FEATURES:
    fc[c] = fc[c].astype("float64")
print("forecast rows:", len(fc))

fg = fs.get_or_create_feature_group(
    name="airqforecast_in963ee7",
    version=1,
    description="Forecast-day weather + lag inputs (pm25 to be predicted)",
    primary_key=["date"],
    event_time="event_time",
    online_enabled=True,
    stream=True,
    features=[
        Feature("date", "string", description="Forecast day (record key)"),
        Feature("event_time", "timestamp", description="Forecast day"),
        Feature("pm25_lag1", "double", description="Previous day's measured pm25"),
        Feature("temperature", "double", description="Daily mean temperature (C)"),
        Feature("humidity", "double", description="Relative humidity (%)"),
        Feature("wind_speed", "double", description="Wind speed"),
        Feature("pressure", "double", description="Atmospheric pressure (hPa)"),
        Feature("precipitation", "double", description="Precipitation (mm)"),
    ],
)
fg.insert(fc, wait=True)
print("inserted forecast inputs into airqforecast_in963ee7")
