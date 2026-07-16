import hopsworks
from hopsworks import udf

proj = hopsworks.login()
fs = proj.get_feature_store()


@udf(float, drop=["request_lat", "request_lon", "home_lat", "home_lon"])
def distance_deg(request_lat, request_lon, home_lat, home_lon):
    dist = ((request_lat - home_lat) ** 2 + (request_lon - home_lon) ** 2) ** 0.5
    return dist.round(6)


@udf(float, drop=["request_lat", "request_lon", "home_lat", "home_lon", "base_score"])
def score(request_lat, request_lon, home_lat, home_lon, base_score):
    dist = (((request_lat - home_lat) ** 2 + (request_lon - home_lon) ** 2) ** 0.5).round(6)
    return (base_score - 0.1 * dist).round(6)


print("distance_deg output names:", distance_deg.output_column_names)
print("distance_deg transformation_features:", distance_deg.transformation_features)
print("score output names:", score.output_column_names)
print("score transformation_features:", score.transformation_features)
print("attrs:", [a for a in dir(distance_deg) if not a.startswith("_")])
