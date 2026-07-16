import pandas as pd

# Load the data
training_df = pd.read_csv('data/training_sample.csv')
serving_df = pd.read_csv('data/serving_log.csv')

# Merge on entity_id
merged_df = pd.merge(training_df, serving_df, on='entity_id', suffixes=('_train', '_serve'))

# Check if serving f2 is training f1 * some constant
print("=== Checking if serving f2 = training f1 * k ===")

# Calculate the ratio for each entity
merged_df['ratio'] = merged_df['f2_serve'] / merged_df['f1_train']

print("Ratio statistics:")
print(merged_df['ratio'].describe())

# Check if the ratio is consistent
print(f"\nRatio mean: {merged_df['ratio'].mean():.6f}")
print(f"Ratio std: {merged_df['ratio'].std():.6f}")

# If the ratio is very consistent, then serving f2 = training f1 * constant
if merged_df['ratio'].std() < 0.01:
    print(f"\n*** CONFIRMED: serving f2 = training f1 * {merged_df['ratio'].mean():.6f} ***")
else:
    print("\nRatio is not consistent enough to be a simple scaling")

# Let's check a few specific examples
print("\n=== Sample ratios ===")
for entity in ['E0001', 'E0002', 'E0003', 'E0004', 'E0005', 'E0007', 'E0008']:
    row = merged_df[merged_df['entity_id'] == entity].iloc[0]
    ratio = row['f2_serve'] / row['f1_train']
    print(f"{entity}: f1_train={row['f1_train']:.6f}, f2_serve={row['f2_serve']:.6f}, ratio={ratio:.6f}")

# Check if serving f2 is actually training f1 * 0.728 or some other constant
# Let's try to find the best fit
import numpy as np

# Try to find k such that f2_serve ≈ f1_train * k
k_values = merged_df['f2_serve'] / merged_df['f1_train']
best_k = k_values.median()
print(f"\nBest fit k (median): {best_k:.6f}")

# Calculate residuals
residuals = merged_df['f2_serve'] - merged_df['f1_train'] * best_k
print(f"Mean residual: {residuals.abs().mean():.6f}")
print(f"Max residual: {residuals.abs().max():.6f}")

# Check if it's actually f1_train * 0.728
k_test = 0.728
residuals_test = merged_df['f2_serve'] - merged_df['f1_train'] * k_test
print(f"\nWith k=0.728:")
print(f"  Mean residual: {residuals_test.abs().mean():.6f}")
print(f"  Max residual: {residuals_test.abs().max():.6f}")

# Actually, let me check if serving f2 is training f1 * 0.728
# But wait, for E0001: 10.938010 * 0.728 = 7.97, but serving f2 = 8.037063
# That's close! Let me check more carefully

print("\n=== Checking if serving f2 = training f1 * 0.728 ===")
for entity in ['E0001', 'E0002', 'E0003', 'E0004', 'E0005']:
    row = merged_df[merged_df['entity_id'] == entity].iloc[0]
    predicted = row['f1_train'] * 0.728
    actual = row['f2_serve']
    diff = abs(predicted - actual)
    print(f"{entity}: predicted={predicted:.6f}, actual={actual:.6f}, diff={diff:.6f}")

# Hmm, that doesn't seem right. Let me think differently.
# Maybe serving f2 is actually training f1 * 0.728 + something?

# Actually, let me just check what the actual relationship is
# Looking at the data again:
# E0001: f1_train=10.938010, f2_serve=8.037063, ratio=0.7347
# E0002: f1_train=7.856750, f2_serve=9.234575, ratio=1.1754
# These ratios are NOT consistent!

# So it's not a simple scaling. Let me check if serving f2 values are actually from training f1
# but for DIFFERENT entities

# Actually, wait. Let me re-examine. The serving log has 200 entities, training has 300.
# Maybe the serving f2 is using training f1 from a different entity?

# Let me check if serving f2 values appear anywhere in training f1
print("\n=== Checking if serving f2 values exist in training f1 ===")
training_f1_set = set(training_df['f1'].values)
serving_f2_set = set(merged_df['f2_serve'].values)

common = training_f1_set & serving_f2_set
print(f"Number of serving f2 values that exist in training f1: {len(common)}")
print(f"Total unique serving f2 values: {len(serving_f2_set)}")

if common:
    print(f"Examples: {list(common)[:10]}")
