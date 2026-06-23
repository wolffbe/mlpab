import warnings; warnings.filterwarnings('ignore')
import json, time
import pandas as pd
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_feature_group('drift_features', version=1)

df = pd.read_csv('data/features.csv')
df['event_time'] = pd.to_datetime(df['event_time'])
days = sorted(df['event_time'].dt.date.unique())
feats = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

records = {}  # date -> {feat: {mean,stddev}}

def capture(day):
    st = fg.statistics
    rec = {}
    for c in st.feature_descriptive_statistics:
        if c.feature_name in feats:
            rec[c.feature_name] = {'mean': c.mean, 'stddev': c.stddev, 'count': c.count}
    records[str(day)] = rec

# day 1 already inserted -> capture its current stats
capture(days[0])

for day in days[1:]:
    sub = df[df['event_time'].dt.date == day]
    fg.insert(sub, write_options={'wait_for_job': True})
    capture(day)
    print('done', day, 'n=', len(sub), flush=True)

with open('day_stats.json', 'w') as f:
    json.dump(records, f, indent=2)
print('SAVED', len(records), 'days')
