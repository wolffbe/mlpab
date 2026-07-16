import hopsworks
import math

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Get the feature group
fg = fs.get_feature_group("leakage_analysis", version=1)

# Read data as dataframe
df = fg.read()

# Compute Pearson correlation between each feature and label
features = ["f1", "f2", "f3", "f4", "f5", "f6"]
n = len(df)

correlations = {}
for feat in features:
    x = df[feat].tolist()
    y = df["label"].tolist()

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if std_x > 0 and std_y > 0:
        corr = cov / (std_x * std_y)
    else:
        corr = 0.0

    correlations[feat] = abs(corr)
    print(f"{feat}: correlation with label = {corr:.6f}, abs = {abs(corr):.6f}")

# Find the feature with highest absolute correlation
best_feature = max(correlations, key=lambda k: correlations[k])
print(f"\nFeature with highest correlation: {best_feature} = {correlations[best_feature]:.6f}")

# Also compute means per label group
print("\nMeans per label group:")
for feat in features:
    group0 = df[df["label"] == 0][feat]
    group1 = df[df["label"] == 1][feat]
    print(f"{feat}: label=0 mean={group0.mean():.4f}, label=1 mean={group1.mean():.4f}, diff={abs(group0.mean() - group1.mean()):.4f}")

# Write result to file
import os
os.makedirs("/srv/hops/hadoop/user/mlpab3ed45e/results", exist_ok=True)
with open("/srv/hops/hadoop/user/mlpab3ed45e/results/leakage_result.txt", "w") as f:
    f.write(f"Best feature (highest |correlation|): {best_feature}\n")
    for feat, corr in sorted(correlations.items(), key=lambda x: -x[1]):
        f.write(f"{feat}: {corr:.6f}\n")

print(f"\nResult: leaked feature is most likely '{best_feature}'")
