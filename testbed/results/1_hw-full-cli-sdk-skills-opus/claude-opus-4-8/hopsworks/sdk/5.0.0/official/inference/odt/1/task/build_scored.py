import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

# --- On-demand feature computed ON THE PLATFORM (RonDB SQL: join + arithmetic) ---
# distance_deg = round(sqrt((req_lat-home_lat)^2 + (req_lon-home_lon)^2), 6)
# score        = round(base_score - 0.1 * distance_deg, 6)   (uses the rounded distance)
sql = """
SELECT request_id,
       account_id,
       distance_deg,
       ROUND(base_score - 0.1 * distance_deg, 6) AS score
FROM (
  SELECT r.request_id  AS request_id,
         r.account_id   AS account_id,
         p.base_score   AS base_score,
         ROUND(SQRT(POWER(r.request_lat - p.home_lat, 2)
                  + POWER(r.request_lon - p.home_lon, 2)), 6) AS distance_deg
  FROM requests_raw_1 r
  JOIN profiles_raw_1 p ON r.account_id = p.account_id
) t
"""
scored = fs.sql(sql, online=True)
scored = scored[["request_id", "account_id", "distance_deg", "score"]]
print("computed rows:", scored.shape)
print("columns:", list(scored.columns))
print("dtypes:\n", scored.dtypes)
print("nulls:\n", scored.isna().sum().to_dict())
print("sample:", scored.head(3).to_dict("records"))
assert scored["request_id"].is_unique, "request_id not unique"
assert scored.shape[0] == 400, "expected 400 rows"

from hsfs.feature import Feature

scored_fg = fs.get_or_create_feature_group(
    name="scored26cb88",
    version=1,
    description="On-demand distance/score feature per scoring request (request coords vs stored profile home).",
    primary_key=["request_id"],
    online_enabled=True,
    stream=True,
    statistics_config=False,
    features=[
        Feature("request_id", "string", description="Scoring request id (record key)"),
        Feature("account_id", "string", description="Account id of the request"),
        Feature("distance_deg", "double", description="round(sqrt((req_lat-home_lat)^2+(req_lon-home_lon)^2),6)"),
        Feature("score", "double", description="round(base_score - 0.1*distance_deg, 6)"),
    ],
)
scored_fg.insert(scored, wait=True)
print("scored26cb88 inserted, id:", scored_fg.id)
print("DONE build")
