"""Pipeline job: compute on-demand distance features and build scored72af4e.

Joins scoring requests (request-time coordinates) with stored account
profiles, applies the on-demand transformation:
  distance_deg = sqrt((request_lat - home_lat)^2 + (request_lon - home_lon)^2)  (rounded to 6 dp)
  score        = base_score - 0.1 * distance_deg                                 (rounded to 6 dp)
and writes the result to feature group scored72af4e v1 (online enabled).
"""

import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()
ds_api = project.get_dataset_api()

req_path = ds_api.download("Resources/scored72af4e/requests.csv", overwrite=True)
prof_path = ds_api.download("Resources/scored72af4e/profiles.csv", overwrite=True)

requests = pd.read_csv(req_path)
profiles = pd.read_csv(prof_path)

df = requests.merge(profiles, on="account_id", how="left")
df["distance_deg"] = (
    (df["request_lat"] - df["home_lat"]) ** 2
    + (df["request_lon"] - df["home_lon"]) ** 2
) ** 0.5
df["distance_deg"] = df["distance_deg"].round(6)
df["score"] = (df["base_score"] - 0.1 * df["distance_deg"]).round(6)

out = df[["request_id", "account_id", "distance_deg", "score"]].copy()
out["request_id"] = out["request_id"].astype(str)
out["account_id"] = out["account_id"].astype(str)

assert len(out) == len(requests), "row count mismatch"
assert not out["distance_deg"].isna().any(), "missing profile join"

fg = fs.get_or_create_feature_group(
    name="scored72af4e",
    version=1,
    primary_key=["request_id"],
    online_enabled=True,
    description="Scored requests: on-demand distance_deg and score per request",
)
fg.insert(out, wait=True)
print(f"inserted {len(out)} rows into scored72af4e v1")
