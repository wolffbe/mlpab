import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

# --- Load source CSVs (ingestion only; no joins/transforms done locally) ---
req = pd.read_csv("data/requests.csv")
prof = pd.read_csv("data/profiles.csv")
req["requested_at"] = pd.to_datetime(req["requested_at"], utc=True)
print("requests:", req.shape, list(req.columns))
print("profiles:", prof.shape, list(prof.columns))

# --- requests_raw FG ---
req_fg = fs.get_or_create_feature_group(
    name="requests_raw",
    version=1,
    description="Raw scoring requests with request-time coordinates",
    primary_key=["request_id"],
    event_time="requested_at",
    online_enabled=True,
    stream=True,
    statistics_config=False,
)
req_fg.insert(req, wait=True)
print("requests_raw inserted, id:", req_fg.id)

# --- profiles_raw FG ---
prof_fg = fs.get_or_create_feature_group(
    name="profiles_raw",
    version=1,
    description="Stored account profiles (home coords + base score)",
    primary_key=["account_id"],
    online_enabled=True,
    stream=True,
    statistics_config=False,
)
prof_fg.insert(prof, wait=True)
print("profiles_raw inserted, id:", prof_fg.id)

print("DONE ingest")
