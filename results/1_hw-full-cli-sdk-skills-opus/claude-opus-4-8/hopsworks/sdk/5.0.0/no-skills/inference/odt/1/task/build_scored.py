import hopsworks
from hopsworks import udf

proj = hopsworks.login()
fs = proj.get_feature_store()

# --- on-demand transformations: computed at request time from request coords + stored profile ---
@udf(float, drop=["request_lat", "request_lon", "home_lat", "home_lon"])
def distance_deg(request_lat, request_lon, home_lat, home_lon):
    dist = ((request_lat - home_lat) ** 2 + (request_lon - home_lon) ** 2) ** 0.5
    return dist.round(6)


@udf(float, drop=["request_lat", "request_lon", "home_lat", "home_lon", "base_score"])
def score(request_lat, request_lon, home_lat, home_lon, base_score):
    dist = (((request_lat - home_lat) ** 2 + (request_lon - home_lon) ** 2) ** 0.5).round(6)
    return (base_score - 0.1 * dist).round(6)


dist_t = distance_deg.alias("distance_deg")
score_t = score.alias("score")

# --- platform-side join of requests + profiles ---
req_fg = fs.get_feature_group("scored26cb88_requests", version=1)
prof_fg = fs.get_feature_group("scored26cb88_profiles", version=1)
q = req_fg.select(["request_id", "account_id", "request_lat", "request_lon"]).join(
    prof_fg.select(["home_lat", "home_lon", "base_score"]), on=["account_id"]
)
joined = q.read()
print("joined shape:", joined.shape, joined.columns.tolist())

# --- target feature group with on-demand transformations, online enabled ---
scored = fs.create_feature_group(
    name="scored26cb88",
    version=1,
    description="Request-time scored features (distance_deg, score) via on-demand transformation",
    primary_key=["request_id"],
    online_enabled=True,
    transformation_functions=[dist_t, score_t],
)
scored.insert(joined, wait=True)
print("inserted into scored26cb88")
print("schema:", [f.name for f in scored.features])
