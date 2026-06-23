import hopsworks, pandas as pd
proj = hopsworks.login()
fs = proj.get_feature_store()
req_fg = fs.get_feature_group('scored26cb88_requests', version=1)
prof_fg = fs.get_feature_group('scored26cb88_profiles', version=1)
q = req_fg.select(['request_id', 'account_id', 'request_lat', 'request_lon']).join(
    prof_fg.select(['home_lat', 'home_lon', 'base_score']), on=['account_id'])
print('reading join...')
df = q.read()
print(df.shape)
print(df.columns.tolist())
print(df.head().to_string())
