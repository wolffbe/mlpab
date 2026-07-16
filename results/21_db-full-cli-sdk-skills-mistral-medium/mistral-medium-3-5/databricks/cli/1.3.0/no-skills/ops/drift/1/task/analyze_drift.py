#!/usr/bin/env python3
"""
Drift detection script to run on Databricks.
Reads data from workspace and detects which feature drifted and when.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import os

# Path to the data file in workspace
DATA_PATH = '/Workspace/Users/benedict@logicalclocks.com/mlpab7b8855/features.csv'
OUTPUT_PATH = '/Workspace/Users/benedict@logicalclocks.com/mlpab7b8855/answers.json'

def main():
    # Read the data
    df = pd.read_csv(DATA_PATH)
    
    # Parse event_time
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['date'] = df['event_time'].dt.date
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
    
    # Method 1: Split into two halves and test for drift
    print("\n" + "="*60)
    print("Method 1: Two-sample tests (first half vs second half)")
    print("="*60)
    
    halfway = len(df) // 2
    first_half = df.iloc[:halfway]
    second_half = df.iloc[halfway:]
    
    drifted_features = []
    for feature in features:
        group1 = first_half[feature].dropna()
        group2 = second_half[feature].dropna()
        
        # Kolmogorov-Smirnov test (non-parametric, detects distribution changes)
        ks_stat, ks_pvalue = stats.ks_2samp(group1, group2)
        
        # T-test (detects mean changes)
        t_stat, t_pvalue = stats.ttest_ind(group1, group2)
        
        print(f"\n{feature}:")
        print(f"  KS: stat={ks_stat:.4f}, p-value={ks_pvalue:.6f}")
        print(f"  T-test: stat={t_stat:.4f}, p-value={t_pvalue:.6f}")
        print(f"  First half: mean={group1.mean():.4f}, std={group1.std():.4f}")
        print(f"  Second half: mean={group2.mean():.4f}, std={group2.std():.4f}")
        
        if ks_pvalue < 0.05:
            drifted_features.append(feature)
            print(f"  *** DRIFT DETECTED (KS test) ***")
    
    print(f"\nDrifted features (KS test): {drifted_features}")
    
    # Method 2: Find the onset date for the drifted feature
    if len(drifted_features) == 1:
        drifted_feature = drifted_features[0]
        print(f"\n" + "="*60)
        print(f"Finding onset date for {drifted_feature}")
        print("="*60)
        
        # Get daily means
        daily_data = df.groupby('date')[drifted_feature].agg(['mean', 'std', 'count']).reset_index()
        daily_data = daily_data.sort_values('date')
        
        # Find the date with the largest sustained change
        # Use a sliding window approach
        dates = sorted(df['date'].unique())
        
        # Calculate mean before and after each date
        best_onset = None
        best_diff = 0
        
        for i in range(1, len(dates) - 1):
            test_date = dates[i]
            
            # Data before test_date
            before = df[df['date'] < test_date][drifted_feature]
            # Data after test_date (including test_date)
            after = df[df['date'] >= test_date][drifted_feature]
            
            if len(before) > 10 and len(after) > 10:  # Need enough samples
                before_mean = before.mean()
                after_mean = after.mean()
                diff = abs(after_mean - before_mean)
                
                # Also check statistical significance
                t_stat, p_value = stats.ttest_ind(before, after)
                
                if diff > best_diff and p_value < 0.05:
                    best_diff = diff
                    best_onset = test_date
                    print(f"  Candidate: {test_date}, diff={diff:.4f}, p-value={p_value:.6f}")
        
        print(f"\nBest onset date: {best_onset}")
        print(f"Mean difference: {best_diff:.4f}")
        
        # Verify by checking the daily means around the onset
        print(f"\nDaily means around onset:")
        for i in range(max(0, dates.index(best_onset) - 3), min(len(dates), dates.index(best_onset) + 4)):
            date = dates[i]
            mean_val = daily_data[daily_data['date'] == date]['mean'].values[0]
            print(f"  {date}: {mean_val:.4f}")
        
        # Write the answer
        answer = {
            "feature": drifted_feature,
            "onset": str(best_onset)
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(answer, f)
        
        print(f"\nAnswer written to {OUTPUT_PATH}")
        print(f"Answer: {json.dumps(answer, indent=2)}")
        
        # Also write to submission directory
        SUBMISSION_PATH = '/Workspace/Users/benedict@logicalclocks.com/mlpab7b8855/submission/answers.json'
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        with open(SUBMISSION_PATH, 'w') as f:
            json.dump(answer, f)
        print(f"Answer also written to {SUBMISSION_PATH}")
        
    elif len(drifted_features) == 0:
        print("\nNo drift detected with KS test. Trying with less strict criteria...")
        # Try with T-test
        for feature in features:
            group1 = first_half[feature].dropna()
            group2 = second_half[feature].dropna()
            t_stat, t_pvalue = stats.ttest_ind(group1, group2)
            if t_pvalue < 0.05:
                drifted_features.append(feature)
                print(f"  {feature} shows drift with T-test (p={t_pvalue:.6f})")
        
        if len(drifted_features) == 1:
            drifted_feature = drifted_features[0]
            print(f"\nUsing T-test: drifted feature = {drifted_feature}")
            # Find onset date
            dates = sorted(df['date'].unique())
            best_onset = None
            best_diff = 0
            
            for i in range(1, len(dates) - 1):
                test_date = dates[i]
                before = df[df['date'] < test_date][drifted_feature]
                after = df[df['date'] >= test_date][drifted_feature]
                
                if len(before) > 10 and len(after) > 10:
                    before_mean = before.mean()
                    after_mean = after.mean()
                    diff = abs(after_mean - before_mean)
                    t_stat, p_value = stats.ttest_ind(before, after)
                    
                    if diff > best_diff and p_value < 0.05:
                        best_diff = diff
                        best_onset = test_date
            
            answer = {
                "feature": drifted_feature,
                "onset": str(best_onset)
            }
            
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(answer, f)
            
            SUBMISSION_PATH = '/Workspace/Users/benedict@logicalclocks.com/mlpab7b8855/submission/answers.json'
            os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
            with open(SUBMISSION_PATH, 'w') as f:
                json.dump(answer, f)
            
            print(f"Answer: {json.dumps(answer, indent=2)}")
        else:
            print(f"\nMultiple or no features detected with T-test: {drifted_features}")
    else:
        print(f"\nMultiple features show drift: {drifted_features}")
        print("This shouldn't happen according to the task description.")

if __name__ == "__main__":
    main()
