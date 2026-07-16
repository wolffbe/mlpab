import pandas as pd

# Load the data
training_df = pd.read_csv('data/training_sample.csv')
serving_df = pd.read_csv('data/serving_log.csv')

# Merge on entity_id
merged_df = pd.merge(training_df, serving_df, on='entity_id', suffixes=('_train', '_serve'))

print("=== Checking if serving f2 equals training f1 ===")
# For each entity, check if serving f2 matches training f1
merged_df['f2_serve_eq_f1_train'] = (merged_df['f2_serve'] == merged_df['f1_train'])
merged_df['f2_serve_close_f1_train'] = (abs(merged_df['f2_serve'] - merged_df['f1_train']) < 0.0001)

print(f"Exact matches: {merged_df['f2_serve_eq_f1_train'].sum()}")
print(f"Close matches: {merged_df['f2_serve_close_f1_train'].sum()}")

# Show some examples where they match
matches = merged_df[merged_df['f2_serve_close_f1_train']][['entity_id', 'f1_train', 'f2_train', 'f2_serve']]
print(f"\nFirst 10 matches:")
print(matches.head(10))

# Check all features in serving against all features in training
print("\n=== Full feature comparison ===")
for serve_feature in ['f1', 'f2', 'f3', 'f4', 'f5']:
    for train_feature in ['f1', 'f2', 'f3', 'f4', 'f5']:
        # Check if they're identical
        identical = (merged_df[f'{serve_feature}_serve'] == merged_df[f'{train_feature}_train']).all()
        if identical:
            print(f"serving {serve_feature} == training {train_feature} *** EXACT MATCH ***")
        else:
            # Check correlation
            corr = merged_df[f'{serve_feature}_serve'].corr(merged_df[f'{train_feature}_train'])
            if corr > 0.999:
                print(f"serving {serve_feature} vs training {train_feature}: corr={corr:.6f} *** NEAR PERFECT ***")

# Let's check the actual values more carefully
print("\n=== Looking at specific entities ===")
# Pick a few entities and check all their values
for entity in ['E0001', 'E0002', 'E0003', 'E0004', 'E0005']:
    row = merged_df[merged_df['entity_id'] == entity].iloc[0]
    print(f"\n{entity}:")
    print(f"  Training: f1={row['f1_train']:.6f}, f2={row['f2_train']:.6f}, f3={row['f3_train']:.6f}, f4={row['f4_train']:.6f}, f5={row['f5_train']:.6f}")
    print(f"  Serving:  f1={row['f1_serve']:.6f}, f2={row['f2_serve']:.6f}, f3={row['f3_serve']:.6f}, f4={row['f4_serve']:.6f}, f5={row['f5_serve']:.6f}")
    
    # Check which serving feature matches which training feature
    for serve_feature in ['f1', 'f2', 'f3', 'f4', 'f5']:
        for train_feature in ['f1', 'f2', 'f3', 'f4', 'f5']:
            if abs(row[f'{serve_feature}_serve'] - row[f'{train_feature}_train']) < 0.0001:
                print(f"    serving {serve_feature} == training {train_feature}")
