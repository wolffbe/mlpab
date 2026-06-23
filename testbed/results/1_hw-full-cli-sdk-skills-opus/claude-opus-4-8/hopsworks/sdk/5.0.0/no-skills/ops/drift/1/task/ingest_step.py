import warnings; warnings.filterwarnings('ignore')
import time, pandas as pd
import hopsworks

t0 = time.time()
proj = hopsworks.login()
fs = proj.get_feature_store()

df = pd.read_csv('data/features.csv')
df['event_time'] = pd.to_datetime(df['event_time'])
print('rows', len(df), 'days', df['event_time'].dt.date.nunique())

fg = fs.create_feature_group(
    name='drift_features',
    version=1,
    description='daily feature observations for drift detection',
    primary_key=['entity_id', 'event_time'],
    event_time='event_time',
    time_travel_format='DELTA',
    online_enabled=False,
    statistics_config={'enabled': True, 'histograms': True, 'correlations': False},
)
d1 = df[df['event_time'].dt.date == pd.Timestamp('2026-01-01').date()]
print('inserting day1 rows', len(d1))
t1 = time.time()
fg.insert(d1, write_options={'wait_for_job': True})
print('insert+job took', round(time.time() - t1, 1), 's')
print('total', round(time.time() - t0, 1), 's')
