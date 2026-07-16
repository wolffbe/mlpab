#!/usr/bin/env python3
"""
Feature engineering script for credit card fraud detection.
Ingests data/transactions.csv, engineers features, and writes to feature group cctxnc444ca.
"""

import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def main():
    # Connect to Hopsworks
    import hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()
    dataset_api.download("Resources/data/transactions.csv", overwrite=True)
    transactions = pd.read_csv('transactions.csv', parse_dates=['datetime'])

    # Feature 1: Transaction velocity (count per card in last 1 hour)
    transactions = transactions.set_index('datetime')
    transactions['transaction_velocity'] = transactions.groupby('cc_num')['transaction_id'].transform(
        lambda x: x.rolling('1H').count()
    )

    # Feature 2: Geo distance from previous transaction (km)
    def calculate_geo_distance(group):
        group = group.sort_values('datetime')
        prev_lat = group['lat'].shift(1)
        prev_long = group['long'].shift(1)
        group['geo_distance'] = group.apply(
            lambda row: np.sqrt((row['lat'] - prev_lat[row.name])**2 + (row['long'] - prev_long[row.name])**2)
            if pd.notna(prev_lat[row.name]) else 0,
            axis=1
        )
        return group

    transactions = transactions.groupby('cc_num').apply(calculate_geo_distance).reset_index(drop=True)

    # Feature 3: Amount per hour (rolling sum per card in last 1 hour)
    transactions['amount_per_hour'] = transactions.groupby('cc_num')['amount'].transform(
        lambda x: x.rolling('1H').sum()
    )

    # Write to feature group
    fg = fs.get_feature_group('cctxnc444ca', version=1)
    fg.insert(transactions.reset_index())


if __name__ == "__main__":
    main()