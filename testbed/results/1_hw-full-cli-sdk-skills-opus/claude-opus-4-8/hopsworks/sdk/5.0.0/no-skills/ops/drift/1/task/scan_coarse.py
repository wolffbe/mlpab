import warnings; warnings.filterwarnings('ignore')
import json
import pandas as pd
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()

df = pd.read_csv('data/features.csv')
df['event_time'] = pd.to_datetime(df['event_time'])
df['d'] = df['event_time'].dt.date
days = sorted(df['d'].unique())
feats = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

# 9 chunks of 10 days
chunks = [days[i:i+10] for i in range(0, len(days), 10)]
labels = [f'{c[0]}..{c[-1]}' for c in chunks]

fg = fs.get_or_create_feature_group(
    name='drift_scan_coarse',
    version=1,
    description='coarse 10-day chunk scan for drift',
    primary_key=['entity_id', 'event_time'],
    event_time='event_time',
    time_travel_format='DELTA',
    online_enabled=False,
    statistics_config={'enabled': True, 'histograms': False, 'correlations': False},
)

order = []  # insertion order -> label
for c, lab in zip(chunks, labels):
    sub = df[df['d'].isin(c)].drop(columns=['d'])
    fg.insert(sub, write_options={'wait_for_job': True})
    order.append((lab, len(sub)))
    print('inserted', lab, len(sub), flush=True)

# map commits (sorted by time = insertion order) to labels, then to stats
cd = {int(k): v for k, v in fg.commit_details().items()}
commit_times = sorted(cd.keys())
allstats = fg.get_all_statistics()
by_end = {}
for s in allstats:
    by_end.setdefault(s.window_end_commit_time, s.computation_time)

result = {}
print('\nchunk  ' + '  '.join(feats))
for ct, (lab, n) in zip(commit_times, order):
    comp = by_end.get(ct)
    full = fg.get_statistics(computation_time=comp)
    d = {x.feature_name: (x.mean, x.stddev) for x in full.feature_descriptive_statistics}
    result[lab] = {f: {'mean': d[f][0], 'stddev': d[f][1]} for f in feats}
    print(lab, '  '.join(f'{d[f][0]:.2f}' for f in feats))

with open('coarse_stats.json', 'w') as fp:
    json.dump(result, fp, indent=2, default=str)
print('SAVED coarse_stats.json')
