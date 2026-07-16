#!/usr/bin/env python3

import csv
import json
import os

def main():
    # Read and filter the CSV data
    valid_categories = {'grocery', 'travel', 'salary', 'rent', 'other'}
    valid_rows = []
    rejected_rows = []
    
    with open('data/events.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row['row_id']
            amount_str = row['amount'].strip()
            category = row['category'].strip()
            
            # Check rule 1: amount is present
            if not amount_str:
                rejected_rows.append(row_id)
                continue
            
            # Check rule 2: amount is within [0, 10000]
            try:
                amount = float(amount_str)
                if amount < 0 or amount > 10000:
                    rejected_rows.append(row_id)
                    continue
            except ValueError:
                rejected_rows.append(row_id)
                continue
            
            # Check rule 3: category is valid
            if category not in valid_categories:
                rejected_rows.append(row_id)
                continue
            
            valid_rows.append(row)
    
    # Write the answers.json file
    os.makedirs('submission', exist_ok=True)
    with open('submission/answers.json', 'w') as f:
        json.dump({"rejected": sorted(rejected_rows)}, f)
    
    # Write filtered CSV
    with open('filtered_events.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['row_id', 'account_id', 'event_time', 'amount', 'category'])
        writer.writeheader()
        for row in valid_rows:
            writer.writerow(row)
    
    print(f"Valid rows: {len(valid_rows)}, Rejected: {len(rejected_rows)}")
    print(f"Filtered CSV written to filtered_events.csv")

if __name__ == '__main__':
    main()