#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json

# Read inputs
feature_history = pd.read_csv("data/feature_history.csv")
with open("data/model.json", "r") as f:
    model = json.load(f)

# Filter for most recent revision at or before T
T = 1773496800000
filtered = feature_history[feature_history["event_time"] <= T]
filtered = filtered.sort_values("event_time").groupby("account_id").last().reset_index()

# Compute score
weights = model["weights"]
bias = model["bias"]
filtered["score"] = 1 / (1 + np.exp(-(
    weights["f1"] * filtered["f1"] +
    weights["f2"] * filtered["f2"] +
    weights["f3"] * filtered["f3"] +
    bias
)))
filtered["score"] = filtered["score"].round(6)

# Save results
result = filtered[["account_id", "score"]]
result.to_csv("scores.csv", index=False)