import pandas as pd
import numpy as np

# Load the data
training_df = pd.read_csv('data/training_sample.csv')
serving_df = pd.read_csv('data/serving_log.csv')

print("Training data shape:", training_df.shape)
print("Serving data shape:", serving_df.shape)

# Merge on entity_id to compare
merged_df = pd.merge(training_df, serving_df, on='entity_id', suffixes=('_train', '_serve'))

print(f"\nMerged data shape: {merged_df.shape}")
print(f"Number of common entities: {len(merged_df)}")

# Calculate differences for each feature
features = ['f1', 'f2', 'f3', 'f4', 'f5']

print("\n=== Feature Comparison ===")
for feature in features:
    train_col = f'{feature}_train'
    serve_col = f'{feature}_serve'
    
    # Calculate statistics
    train_mean = merged_df[train_col].mean()
    serve_mean = merged_df[serve_col].mean()
    train_std = merged_df[train_col].std()
    serve_std = merged_df[serve_col].std()
    
    # Calculate absolute differences
    diffs = (merged_df[serve_col] - merged_df[train_col]).abs()
    mean_diff = diffs.mean()
    max_diff = diffs.max()
    
    print(f"\n{feature}:")
    print(f"  Training mean: {train_mean:.4f}, std: {train_std:.4f}")
    print(f"  Serving mean: {serve_mean:.4f}, std: {serve_std:.4f}")
    print(f"  Mean absolute difference: {mean_diff:.4f}")
    print(f"  Max absolute difference: {max_diff:.4f}")
    
    # Check if the feature is computed differently
    # If serving values are systematically different from training values
    correlation = merged_df[[train_col, serve_col]].corr().iloc[0, 1]
    print(f"  Correlation between train and serve: {correlation:.4f}")

# Let's also check if any feature has a consistent multiplier or offset
print("\n=== Checking for systematic transformations ===")
for feature in features:
    train_col = f'{feature}_train'
    serve_col = f'{feature}_serve'
    
    # Check if serving = train * constant
    ratios = merged_df[serve_col] / merged_df[train_col]
    ratio_mean = ratios.mean()
    ratio_std = ratios.std()
    
    # Check if serving = train + constant
    diffs = merged_df[serve_col] - merged_df[train_col]
    diff_mean = diffs.mean()
    diff_std = diffs.std()
    
    print(f"\n{feature}:")
    print(f"  Ratio (serve/train) mean: {ratio_mean:.4f}, std: {ratio_std:.4f}")
    print(f"  Difference (serve-train) mean: {diff_mean:.4f}, std: {diff_std:.4f}")
    
    # If ratio is consistent, it's likely a scaling issue
    if ratio_std < 0.1 and abs(ratio_mean - 1.0) > 0.1:
        print(f"  *** POTENTIAL SKEW: Consistent ratio of {ratio_mean:.4f}")
    
    # If difference is consistent, it's likely an offset issue
    if diff_std < 0.5 and abs(diff_mean) > 0.5:
        print(f"  *** POTENTIAL SKEW: Consistent difference of {diff_mean:.4f}")

# Let's look at the actual values for the most suspicious feature
print("\n=== Detailed comparison for f2 ===")
f2_comparison = merged_df[['entity_id', 'f2_train', 'f2_serve']].copy()
f2_comparison['f2_ratio'] = f2_comparison['f2_serve'] / f2_comparison['f2_train']
f2_comparison['f2_diff'] = f2_comparison['f2_serve'] - f2_comparison['f2_train']

print(f2_comparison.head(20))
print(f"\nF2 ratio statistics:")
print(f2_comparison['f2_ratio'].describe())

# Check if f2 in serving is consistently different
print("\n=== Checking if f2 is systematically different ===")
# Let's see if there's a pattern - maybe f2 in serving is actually from training's f1 or another feature

# Check correlation between serving f2 and training f1
corr_f2serve_f1train = merged_df['f2_serve'].corr(merged_df['f1_train'])
print(f"Correlation between serving f2 and training f1: {corr_f2serve_f1train:.4f}")

# Check if serving f2 matches training f1
f2_serve_vs_f1_train = (merged_df['f2_serve'] - merged_df['f1_train']).abs()
print(f"Mean abs diff between serving f2 and training f1: {f2_serve_vs_f1_train.mean():.4f}")
print(f"Max abs diff between serving f2 and training f1: {f2_serve_vs_f1_train.max():.4f}")

# Check all pairwise correlations
print("\n=== All pairwise correlations between serving f2 and training features ===")
for train_feature in features:
    corr = merged_df['f2_serve'].corr(merged_df[f'{train_feature}_train'])
    print(f"  serving f2 vs training {train_feature}: {corr:.4f}")

# Let's check if serving f2 is actually training f1 * some constant
print("\n=== Checking if serving f2 = training f1 * constant ===")
ratios_f2serve_f1train = merged_df['f2_serve'] / merged_df['f1_train']
print(f"Ratio statistics: mean={ratios_f2serve_f1train.mean():.4f}, std={ratios_f2serve_f1train.std():.4f}")

# Check other features too
print("\n=== Checking all serving features against training features ===")
for serve_feature in features:
    for train_feature in features:
        if serve_feature == train_feature:
            continue
        corr = merged_df[f'{serve_feature}_serve'].corr(merged_df[f'{train_feature}_train'])
        if corr > 0.9:
            print(f"  serving {serve_feature} vs training {train_feature}: {corr:.4f} ***")
