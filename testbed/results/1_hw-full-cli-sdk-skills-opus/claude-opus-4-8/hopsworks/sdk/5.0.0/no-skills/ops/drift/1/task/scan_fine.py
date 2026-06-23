import warnings; warnings.filterwarnings('ignore')
import json
import pandas as pd
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()

df = pd.read_csv('data/features.csv')
df['event_time'] = pd.to_datetime(df['event_time'])
df['d'] = df['event_time'].dt.date
feats = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

start = pd.Timestamp('2026-03-02').date()
end = pd.Timestamp('2026-03-14').date()
fine_days = [d for d in sorted(df['d'].unique()) if start <= d <= end]

fg = fs.get_or_create_feature_group(
    name='drift_scan_fine',
    version=1,
    description='per-day fine scan around onset',
    primary_key=['entity_id', 'event_time'],
    event_time='event_time',
    time_travel_format='DELTA',
    online_enabled=False,
    statistics_config={'enabled': True, 'histograms': False, 'correlations': False},
)

order = []
for day in fine_days:
    sub = df[df['d'] == day].drop(columns=['d'])
    fg.insert(sub, write_options={'wait_for_job': True})
    order.append(str(day))
    print('inserted', day, len(sub), flush=True)

cd = {int(k): v for k, v in fg.commit_details().items()}
commit_times = sorted(cd.keys())
allstats = fg.get_all_statistics()
by_end = {}
for s in allstats:
    by_end.setdefault(s.window_end_commit_time, s.computation_time)

result = {}
print('\nday  f3_mean  f3_stddev')
for ct, lab in zip(commit_times, order):
    comp = by_end.get(ct)
    full = fg.get_statistics(computation_time=comp)
    d = {x.feature_name: (x.mean, x.stddev) for x in full.feature_descriptive_statistics}
    result[lab] = {f: {'mean': d[f][0], 'stddev': d[f][1]} for f in feats}
    print(lab, f'{d["f3"][0]:.3f}', f'{d["f3"][1]:.3f}')

with open('fine_stats.json', 'w') as fp:
    json.dump(result, fp, indent=2, default=str)
print('SAVED fine_stats.json')
