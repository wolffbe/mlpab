import hopsworks
import json
import os

# Connect to hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read data files using hopsworks dataset API
dataset_api = project.get_dataset_api()

# Upload the CSV files to hopsworks
dataset_api.upload("data/training_sample.csv", "Resources/skew_analysis", overwrite=True)
dataset_api.upload("data/serving_log.csv", "Resources/skew_analysis", overwrite=True)

# Use pandas via hopsworks context to load data
import pandas as pd

train_df = pd.read_csv("data/training_sample.csv")
serving_df = pd.read_csv("data/serving_log.csv")

features = [c for c in train_df.columns if c != 'entity_id']

print("Feature statistics comparison:")
print(f"{'Feature':8s}  {'Train mean':>12s}  {'Serving mean':>13s}  {'Ratio':>8s}  {'Train std':>10s}  {'Serving std':>12s}")

results = {}
for f in features:
    tm = train_df[f].mean()
    ts = train_df[f].std()
    sm = serving_df[f].mean()
    ss = serving_df[f].std()
    ratio = sm / tm if tm != 0 else float('inf')
    results[f] = {'train_mean': tm, 'serving_mean': sm, 'ratio': ratio, 'train_std': ts, 'serving_std': ss}
    print(f"{f:8s}  {tm:>12.4f}  {sm:>13.4f}  {ratio:>8.4f}  {ts:>10.4f}  {ss:>12.4f}")

# Find the most divergent feature
# Check both mean ratio and std ratio
divergences = {}
for f in features:
    r = results[f]
    mean_div = abs(r['ratio'] - 1.0)
    std_ratio = r['serving_std'] / r['train_std'] if r['train_std'] != 0 else float('inf')
    std_div = abs(std_ratio - 1.0)
    divergences[f] = max(mean_div, std_div)
    print(f"{f}: mean_div={mean_div:.4f}, std_div={std_div:.4f}")

skewed_feature = max(divergences, key=lambda x: divergences[x])
print(f"\nMost divergent feature: {skewed_feature} (divergence score: {divergences[skewed_feature]:.4f})")

# Write submission
os.makedirs("submission", exist_ok=True)
answer = {"feature": skewed_feature, "cause": f"Feature {skewed_feature} shows the largest divergence between training (mean={results[skewed_feature]['train_mean']:.4f}) and serving (mean={results[skewed_feature]['serving_mean']:.4f}), ratio={results[skewed_feature]['ratio']:.4f}"}
with open("submission/answers.json", "w") as f:
    json.dump(answer, f, indent=2)
print(f"\nWritten to submission/answers.json: {answer}")
