import warnings; warnings.filterwarnings('ignore')
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_feature_group('drift_features', version=1)

feats = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
cd = fg.commit_details()
commit_times = sorted(int(k) for k in cd.keys())

print('commit_idx committedOn  ' + '  '.join(f'{f}_mean' for f in feats))
for i, ct in enumerate(commit_times):
    st = fg.get_statistics_by_commit_window(from_commit_time=ct, to_commit_time=ct)
    if st is None:
        print(i, ct, 'NO STATS')
        continue
    stats = st if isinstance(st, list) else [st]
    # take first
    s = stats[0]
    d = {c.feature_name: c.mean for c in s.feature_descriptive_statistics}
    row = '  '.join(f'{d.get(f, float("nan")):.3f}' for f in feats)
    print(i, cd[str(ct)]['committedOn'], row)
