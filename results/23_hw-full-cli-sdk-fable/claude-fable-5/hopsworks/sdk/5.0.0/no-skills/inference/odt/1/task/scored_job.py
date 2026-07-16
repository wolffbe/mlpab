import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()
ds_api = project.get_dataset_api()

req = pd.read_csv("/hopsfs/Resources/requests.csv")
prof = pd.read_csv("/hopsfs/Resources/profiles.csv")
print("requests:", req.shape, "profiles:", prof.shape, flush=True)

req_fg = fs.get_or_create_feature_group(
    name="requests72af4e",
    version=1,
    primary_key=["request_id"],
    description="raw scoring requests",
)
req_fg.insert(req, wait=True)
prof_fg = fs.get_or_create_feature_group(
    name="profiles72af4e",
    version=1,
    primary_key=["account_id"],
    description="stored account profiles",
)
prof_fg.insert(prof, wait=True)
print("raw feature groups ingested", flush=True)

# platform-side join of requests with profiles
try:
    q = req_fg.select(["request_id", "account_id", "request_lat", "request_lon"]).join(
        prof_fg.select(["home_lat", "home_lon", "base_score"]), on=["account_id"]
    )
    joined = q.read()
    assert len(joined) == len(req), f"unexpected join size {len(joined)}"
except Exception as e:
    print("feature store query read failed, merging in-job instead:", e, flush=True)
    joined = req[["request_id", "account_id", "request_lat", "request_lon"]].merge(
        prof, on="account_id", how="left"
    )
print("joined:", joined.shape, flush=True)

from hsfs.hopsworks_udf import udf


@udf(
    return_type=[float, float],
    drop=["request_lat", "request_lon", "home_lat", "home_lon", "base_score"],
    mode="pandas",
)
def dist_score(request_lat, request_lon, home_lat, home_lon, base_score):
    import pandas as pd

    distance_deg = (
        (request_lat - home_lat) ** 2 + (request_lon - home_lon) ** 2
    ) ** 0.5
    distance_deg = distance_deg.round(6)
    score = (base_score - 0.1 * distance_deg).round(6)
    return pd.DataFrame({"distance_deg": distance_deg, "score": score})


scored_fg = fs.get_or_create_feature_group(
    name="scored72af4e",
    version=1,
    primary_key=["request_id"],
    online_enabled=True,
    description="on-demand scored requests",
    transformation_functions=[dist_score.alias("distance_deg", "score")],
)
scored_fg.insert(
    joined[
        [
            "request_id",
            "account_id",
            "request_lat",
            "request_lon",
            "home_lat",
            "home_lon",
            "base_score",
        ]
    ],
    wait=True,
)
print("scored inserted", flush=True)
print("schema:", [f.name for f in scored_fg.features], flush=True)
back = scored_fg.read()
print("readback:", back.shape, list(back.columns), flush=True)
print(back.head().to_string(), flush=True)
