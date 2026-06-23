import hopsworks
import pandas as pd
import numpy as np

proj = hopsworks.login()
fs = proj.get_feature_store()
fg_train = fs.get_feature_group("skew_train", version=1)
fg_serve = fs.get_feature_group("skew_serve", version=1)
df = fg_train.select_all().join(
    fg_serve.select_all(), on=["entity_id"], prefix="srv_").read()
df = df.dropna(subset=["srv_f2"])

t = df["f2"].astype(float)
s = df["srv_f2"].astype(float)
# hypothesis: serving = expm1(training)  <=> training = log1p(serving)
pred = np.expm1(t)
print("residual expm1(train) vs serve  max|err|:", (pred - s).abs().max())
print("residual log1p(serve) vs train  max|err|:", (np.log1p(s) - t).abs().max())
