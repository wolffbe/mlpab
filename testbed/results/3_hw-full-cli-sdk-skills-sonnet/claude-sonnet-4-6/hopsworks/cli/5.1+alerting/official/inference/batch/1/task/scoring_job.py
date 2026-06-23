import os
import math
import csv
import json
import hopsworks

T = 1773219600000

WEIGHTS = {"f1": 0.9945, "f2": 0.8451, "f3": 0.5468}
BIAS = 0.8081


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def main():
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()

    dataset_api.download("Resources/scoring_input/feature_history.csv", local_path="/tmp/feature_history.csv", overwrite=True)
    dataset_api.download("Resources/scoring_input/model.json", local_path="/tmp/model.json", overwrite=True)

    with open("/tmp/model.json") as f:
        model_data = json.load(f)
    weights = model_data["weights"]
    bias = model_data["bias"]

    best = {}
    with open("/tmp/feature_history.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            account_id = row["account_id"]
            et = int(row["event_time"])
            if et > T:
                continue
            if account_id not in best or et > best[account_id]["event_time"]:
                best[account_id] = {
                    "event_time": et,
                    "f1": float(row["f1"]),
                    "f2": float(row["f2"]),
                    "f3": float(row["f3"]),
                }

    rows = []
    for account_id, feat in best.items():
        z = (weights["f1"] * feat["f1"]
             + weights["f2"] * feat["f2"]
             + weights["f3"] * feat["f3"]
             + bias)
        score = round(sigmoid(z), 6)
        rows.append({"account_id": account_id, "score": score})

    print(f"Computed {len(rows)} scores")

    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="scores43f1c2",
        version=1,
        primary_key=["account_id"],
        online_enabled=True,
        description="Batch scores as of T=1773219600000",
    )

    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.head())

    fg.insert(df, write_options={"wait_for_job": True})
    print("Insert complete")


if __name__ == "__main__":
    main()
