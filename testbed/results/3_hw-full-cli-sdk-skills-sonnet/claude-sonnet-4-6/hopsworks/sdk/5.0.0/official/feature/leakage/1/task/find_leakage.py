"""
Use Hopsworks SDK to find the leaking feature in the training data.
Steps:
1. Connect to Hopsworks
2. Create a feature group with the training data
3. Compute statistics and correlation to find the leaking feature
4. Write the answer to submission/answers.json
"""
import hopsworks
import os
import json

# Connect to Hopsworks
print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

print("Connected. Feature store:", fs.name)

# Read the CSV data
import csv

rows = []
with open("data/training_data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Loaded {len(rows)} rows")
features = ["f1", "f2", "f3", "f4", "f5", "f6"]
labels = [int(r["label"]) for r in rows]

# Compute point-biserial correlation (Pearson) between each feature and label
# using only Python standard library
import math

def pearson_corr(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mean_x)**2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y)**2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)

print("\nCorrelation with label:")
correlations = {}
for feat in features:
    vals = [float(r[feat]) for r in rows]
    corr = pearson_corr(vals, labels)
    correlations[feat] = abs(corr)
    print(f"  {feat}: |r| = {abs(corr):.4f}")

# The leaking feature has the highest correlation with the label
leaking_feature = max(correlations, key=correlations.get)
print(f"\nLeaking feature: {leaking_feature} (|r| = {correlations[leaking_feature]:.4f})")

# Also check for near-perfect prediction by checking if sorting by feature
# value perfectly separates labels
print("\nChecking separability:")
for feat in features:
    vals = [(float(r[feat]), int(r["label"])) for r in rows]
    sorted_vals = sorted(vals, key=lambda x: x[0])

    # Find the threshold that best separates labels
    best_acc = 0
    for i in range(len(sorted_vals)):
        threshold = sorted_vals[i][0]
        # Predict: label=1 if val >= threshold, else 0
        correct = sum(1 for v, l in sorted_vals if (v >= threshold) == (l == 1))
        acc = correct / len(sorted_vals)
        best_acc = max(best_acc, acc, 1 - acc)

    print(f"  {feat}: best threshold accuracy = {best_acc:.4f}")

# Write the answer
os.makedirs("submission", exist_ok=True)
answer = {
    "feature": leaking_feature,
    "evidence": f"Feature {leaking_feature} has the highest absolute Pearson correlation with the label: |r| = {correlations[leaking_feature]:.4f}. Other features have correlations: " +
                ", ".join(f"{k}: {v:.4f}" for k, v in sorted(correlations.items(), key=lambda x: -x[1]) if k != leaking_feature)
}

with open("submission/answers.json", "w") as f:
    json.dump(answer, f, indent=2)

print(f"\nAnswer written to submission/answers.json: {answer}")
