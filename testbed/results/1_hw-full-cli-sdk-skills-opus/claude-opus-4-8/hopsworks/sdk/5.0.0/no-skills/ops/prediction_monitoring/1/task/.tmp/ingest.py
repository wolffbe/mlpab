import pandas as pd, hopsworks
df = pd.read_csv('data/prediction_log.csv')
df['ts'] = pd.to_datetime(df['ts'])
df = df.reset_index().rename(columns={'index': 'id'})
df['id'] = df['id'].astype('int64')
df['prediction'] = df['prediction'].astype('float64')
print(df.dtypes.to_dict())

proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_or_create_feature_group(
    name='prediction_log',
    version=1,
    description='Deployed model prediction log',
    primary_key=['id'],
    event_time='ts',
    online_enabled=False,
)
fg.insert(df, write_options={"wait_for_job": True})
print('INSERTED rows:', len(df))
