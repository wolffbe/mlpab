import pandas as pd
import numpy as np

# Load the data
training_df = pd.read_csv('data/training_sample.csv')
serving_df = pd.read_csv('data/serving_log.csv')

# Merge on entity_id
merged_df = pd.merge(training_df, serving_df, on='entity_id', suffixes=('_train', '_serve'))

# Check if serving f2 matches any other training feature
print("=== Checking if serving f2 matches other training features ===")
for train_feature in ['f1', 'f3', 'f4', 'f5']:
    diff = (merged_df['f2_serve'] - merged_df[f'{train_feature}_train']).abs()
    mean_diff = diff.mean()
    max_diff = diff.max()
    corr = merged_df['f2_serve'].corr(merged_df[f'{train_feature}_train'])
    print(f"serving f2 vs training {train_feature}: mean_diff={mean_diff:.4f}, max_diff={max_diff:.4f}, corr={corr:.4f}")

# Check if serving f2 is a transformation of training f1
print("\n=== Checking transformations of training f1 ===")

# Try f1 * 3
f1_times_3 = merged_df['f1_train'] * 3
f2_serve = merged_df['f2_serve']
diff = (f2_serve - f1_times_3).abs()
print(f"f2_serve vs f1_train * 3: mean_diff={diff.mean():.4f}, max_diff={diff.max():.4f}")

# Try f1 * 4
f1_times_4 = merged_df['f1_train'] * 4
diff = (f2_serve - f1_times_4).abs()
print(f"f2_serve vs f1_train * 4: mean_diff={diff.mean():.4f}, max_diff={diff.max():.4f}")

# Try f1 * 2
f1_times_2 = merged_df['f1_train'] * 2
diff = (f2_serve - f1_times_2).abs()
print(f"f2_serve vs f1_train * 2: mean_diff={diff.mean():.4f}, max_diff={diff.max():.4f}")

# Check if serving f2 is actually training f1
print("\n=== Checking if serving f2 = training f1 ===")
diff = (merged_df['f2_serve'] - merged_df['f1_train']).abs()
print(f"Mean diff: {diff.mean():.4f}")
print(f"Max diff: {diff.max():.4f}")
print(f"Number of exact matches: {(diff == 0).sum()}")

# Let's look at a few examples
print("\n=== Sample comparisons ===")
sample = merged_df[['entity_id', 'f1_train', 'f2_train', 'f2_serve']].head(10)
print(sample)

# Check if serving f2 is from a different column in serving
print("\n=== Checking if serving f2 is swapped with another serving feature ===")
# Maybe f2 in serving is actually f1 from serving?
# But we need to check the serving data structure

# Let's check all serving features against all training features
print("\n=== Full correlation matrix ===")
for serve_feature in ['f1', 'f2', 'f3', 'f4', 'f5']:
    for train_feature in ['f1', 'f2', 'f3', 'f4', 'f5']:
        corr = merged_df[f'{serve_feature}_serve'].corr(merged_df[f'{train_feature}_train'])
        if corr > 0.99:
            print(f"serving {serve_feature} vs training {train_feature}: {corr:.6f} *** PERFECT MATCH ***")
        elif corr > 0.9:
            print(f"serving {serve_feature} vs training {train_feature}: {corr:.6f} **")

# Check if serving f2 has any relationship with training features
print("\n=== Detailed analysis of serving f2 ===")
print("Serving f2 statistics:")
print(merged_df['f2_serve'].describe())

print("\nTraining f2 statistics:")
print(merged_df['f2_train'].describe())

# Check if serving f2 is actually the training f1 values
print("\n=== Checking if serving f2 contains training f1 values ===")
# For each entity in serving, check if f2_serve matches f1_train
matches = []
for idx, row in merged_df.iterrows():
    if abs(row['f2_serve'] - row['f1_train']) < 0.0001:
        matches.append(row['entity_id'])

print(f"Entities where serving f2 == training f1: {len(matches)}")
if matches:
    print(f"Examples: {matches[:10]}")

# Check the relationship more carefully
print("\n=== Checking ratio distribution ===")
ratios = merged_df['f2_serve'] / merged_df['f2_train']
print(ratios.describe())

# Check if the ratio is consistent
print(f"\nRatio std: {ratios.std():.4f}")
print(f"Ratio mean: {ratios.mean():.4f}")

# Maybe it's not a simple scaling - let's check if serving f2 is actually from training
# but with a different computation

# Let's check if serving f2 values appear in training f1
print("\n=== Checking if serving f2 values exist in training f1 ===")
training_f1_values = set(training_df['f1'].round(6))
serving_f2_values = set(merged_df['f2_serve'].round(6))
intersection = training_f1_values & serving_f2_values
print(f"Number of unique training f1 values: {len(training_f1_values)}")
print(f"Number of unique serving f2 values: {len(serving_f2_values)}")
print(f"Number of common values: {len(intersection)}")
print(f"Percentage of serving f2 values that exist in training f1: {len(intersection) / len(serving_f2_values) * 100:.2f}%")
