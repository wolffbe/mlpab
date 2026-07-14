#!/usr/bin/env python3
import pandas as pd
import json
import numpy as np

# Load inputs
T = 1773234000000
feature_history = pd.read_csv("data/feature_history.csv")
with open("data/model.json", "r") as f:
    model = json.load(f)

# Filter feature history to retain only the most recent revision at or before T
filtered = feature_history[feature_history["event_time"] <= T]
filtered = filtered.sort_values("event_time", ascending=False).drop_duplicates("account_id")

# Compute scores
weights = model["weights"]
bias = model["bias"]

filtered["score"] = 1 / (1 + np.exp(-(
    weights["f1"] * filtered["f1"] +
    weights["f2"] * filtered["f2"] +
    weights["f3"] * filtered["f3"] +
    bias
)))

# Round to 6 decimal places
filtered["score"] = filtered["score"].round(6)

# Save results
result = filtered[["account_id", "score"]]
result.to_csv("scores.csv", index=False)