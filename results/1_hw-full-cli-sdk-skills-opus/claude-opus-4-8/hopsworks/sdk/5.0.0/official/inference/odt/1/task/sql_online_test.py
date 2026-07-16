import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

sql = """
SELECT r.request_id AS request_id,
       r.account_id AS account_id,
       ROUND(SQRT(POWER(r.request_lat - p.home_lat, 2) + POWER(r.request_lon - p.home_lon, 2)), 6) AS distance_deg
FROM requests_raw_1 r
JOIN profiles_raw_1 p ON r.account_id = p.account_id
LIMIT 5
"""
try:
    df = fs.sql(sql, online=True)
    print("fs.sql online OK")
    print(list(df.columns))
    print(df.to_dict("records"))
except Exception as ex:
    print("fs.sql online ERR:", repr(ex)[:600])
