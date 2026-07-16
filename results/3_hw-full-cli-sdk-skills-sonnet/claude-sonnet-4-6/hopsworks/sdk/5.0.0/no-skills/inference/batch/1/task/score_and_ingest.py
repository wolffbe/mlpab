import csv
import math
import os

# ── 1. Load data ──────────────────────────────────────────────────────────────
T = 1773219600000  # as-of timestamp (epoch ms)

WEIGHTS = {"f1": 0.9945, "f2": 0.8451, "f3": 0.5468}
BIAS = 0.8081

data_path = os.path.join(os.path.dirname(__file__), "data", "feature_history.csv")

# Keep the latest revision per account with event_time <= T
latest = {}  # account_id -> (event_time, f1, f2, f3)

with open(data_path, newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        et = int(row["event_time"])
        if et > T:
            continue
        aid = row["account_id"]
        if aid not in latest or et > latest[aid][0]:
            latest[aid] = (et, float(row["f1"]), float(row["f2"]), float(row["f3"]))

# ── 2. Score ──────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

rows = []
for aid, (_, f1, f2, f3) in sorted(latest.items()):
    logit = WEIGHTS["f1"] * f1 + WEIGHTS["f2"] * f2 + WEIGHTS["f3"] * f3 + BIAS
    score = round(sigmoid(logit), 6)
    rows.append({"account_id": aid, "score": score})

print(f"Accounts scored: {len(rows)}")
print("Sample rows:", rows[:3])

# ── 3. Push to Hopsworks ──────────────────────────────────────────────────────
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

import pandas as pd
df = pd.DataFrame(rows)

fg = fs.get_or_create_feature_group(
    name="scores43f1c2",
    version=1,
    primary_key=["account_id"],
    online_enabled=True,
    description="Batch scores as of T=1773219600000",
)

fg.insert(df, write_options={"wait_for_job": True})

print("Done — feature group scores43f1c2 v1 created and populated.")
