# Databricks notebook source
# MAGIC %md
# MAGIC ## Drift Detection Analysis
# MAGIC 
# MAGIC This notebook analyzes feature drift in the dataset.

# COMMAND ----------

# Read the data
import pandas as pd
from datetime import datetime

# Read from workspace
df = pd.read_csv('/Workspace/Users/benedict@logicalclocks.com/mlpab7b8855/features.csv')

# Parse event_time
df['event_time'] = pd.to_datetime(df['event_time'])
df['date'] = df['event_time'].dt.date

print(f"Data shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Features: {df.columns.tolist()}")

# COMMAND ----------

# Analyze each feature's distribution over time
import numpy as np
import matplotlib.pyplot as plt

features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

# Calculate daily statistics for each feature
stats_by_date = {}
for feature in features:
    daily_stats = df.groupby('date')[feature].agg(['mean', 'std', 'min', 'max']).reset_index()
    stats_by_date[feature] = daily_stats
    
    # Print summary
    print(f"\n{feature}:")
    print(f"  Overall mean: {df[feature].mean():.4f}, std: {df[feature].std():.4f}")
    print(f"  Min: {df[feature].min():.4f}, Max: {df[feature].max():.4f}")

# COMMAND ----------

# Detect drift using statistical tests
from scipy import stats

# Split data into two halves to detect drift
halfway = len(df) // 2
first_half = df.iloc[:halfway]
second_half = df.iloc[halfway:]

print("\n\nDrift detection (first half vs second half):")
print("=" * 60)

for feature in features:
    group1 = first_half[feature].dropna()
    group2 = second_half[feature].dropna()
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(group1, group2)
    
    # T-test
    t_stat, t_pvalue = stats.ttest_ind(group1, group2)
    
    print(f"\n{feature}:")
    print(f"  KS statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.6f}")
    print(f"  T-test statistic: {t_stat:.4f}, p-value: {t_pvalue:.6f}")
    print(f"  First half mean: {group1.mean():.4f}, std: {group1.std():.4f}")
    print(f"  Second half mean: {group2.mean():.4f}, std: {group2.std():.4f}")
    
    if ks_pvalue < 0.05 or t_pvalue < 0.05:
        print(f"  *** DRIFT DETECTED ***")

# COMMAND ----------

# Find the onset date by checking when the distribution changes
print("\n\nFinding onset date for drifted feature:")
print("=" * 60)

# Based on the test above, identify which feature drifted
# Let's check all features more carefully with rolling windows

for feature in features:
    # Calculate rolling mean and std
    df_sorted = df.sort_values('date')
    rolling_mean = df_sorted.groupby('date')[feature].mean()
    
    # Find the date with maximum change in rolling mean
    mean_diff = rolling_mean.diff().abs()
    if len(mean_diff) > 0:
        max_change_date = mean_diff.idxmax()
        max_change = mean_diff.max()
        
        # Also check for sustained shift
        dates = sorted(df['date'].unique())
        mid_date = dates[len(dates) // 2]
        
        before_mean = df[df['date'] < mid_date][feature].mean()
        after_mean = df[df['date'] >= mid_date][feature].mean()
        mean_shift = abs(after_mean - before_mean)
        
        print(f"\n{feature}:")
        print(f"  Max single-day change: {max_change:.4f} on {max_change_date}")
        print(f"  Mean shift at midpoint: {mean_shift:.4f}")
        print(f"  Before {mid_date}: mean={before_mean:.4f}")
        print(f"  After {mid_date}: mean={after_mean:.4f}")

# COMMAND ----------

# More precise: find the exact onset date
print("\n\nPrecise onset detection:")
print("=" * 60)

# For each feature, find when the mean significantly changes
for feature in features:
    dates = sorted(df['date'].unique())
    daily_means = df.groupby('date')[feature].mean()
    
    # Find the date with the largest jump
    max_jump = 0
    onset_date = None
    
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        prev_mean = daily_means[prev_date]
        curr_mean = daily_means[curr_date]
        jump = abs(curr_mean - prev_mean)
        
        if jump > max_jump:
            max_jump = jump
            onset_date = curr_date
    
    print(f"{feature}: max jump = {max_jump:.4f} on {onset_date}")

# COMMAND ----------

# Visual inspection - save plots
import os
output_dir = '/dbfs/FileStore/drift_plots'
os.makedirs(output_dir, exist_ok=True)

for feature in features:
    plt.figure(figsize=(12, 6))
    daily_means = df.groupby('date')[feature].mean()
    daily_means.plot(kind='line', marker='o')
    plt.title(f'{feature} - Daily Mean')
    plt.xlabel('Date')
    plt.ylabel('Mean Value')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{feature}.png')
    plt.close()
    print(f"Saved plot for {feature}")

print("\nAnalysis complete!")
