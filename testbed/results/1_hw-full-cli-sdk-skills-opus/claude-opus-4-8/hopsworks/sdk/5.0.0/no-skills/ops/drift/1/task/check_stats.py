import warnings; warnings.filterwarnings('ignore')
import json
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_feature_group('drift_features', version=1)

st = fg.statistics
print('TYPE', type(st))
print('attrs', [a for a in dir(st) if not a.startswith('_')])
content = st.feature_descriptive_statistics
for c in content:
    print(c.feature_name, 'mean=', c.mean, 'stddev=', c.stddev, 'count=', c.count)
