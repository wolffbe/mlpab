#!/usr/bin/env python3

import hopsworks
import pandas as pd

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read data from Resources
transactions_df = pd.read_csv("Resources/transactions.csv")
fx_rates_df = pd.read_csv("Resources/fx_rates.csv")

# Get feature groups
transactions_fg = fs.get_feature_group("transactions", version=1)
fx_rates_fg = fs.get_feature_group("fx_rates", version=1)

# Ingest data
transactions_fg.insert(transactions_df)
fx_rates_fg.insert(fx_rates_df)