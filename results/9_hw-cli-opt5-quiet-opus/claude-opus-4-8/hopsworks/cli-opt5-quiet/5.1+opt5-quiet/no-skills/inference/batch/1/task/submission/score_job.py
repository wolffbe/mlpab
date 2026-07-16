"""Batch-score every account as-of T and write an online feature group.

Runs as a Hopsworks PYTHON job on the cluster (platform-side compute).
"""
import math

import hopsworks

# --- Model (data/model.json) ---
W_F1 = -0.4502
W_F2 = 0.4813
W_F3 = 0.6773
BIAS = 0.0357

# --- As-of timestamp T (data/scoring_request.md), epoch milliseconds ---
T = 1773590400000


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Read the full feature history from the offline store.
    hist_fg = fs.get_feature_group("feathistb6c522", version=1)
    df = hist_fg.read()

    # Normalise column names / types.
    df = df[["account_id", "event_time", "f1", "f2", "f3"]].copy()
    df["event_time"] = df["event_time"].astype("int64")

    # Point-in-time: keep only revisions valid at or before T, then the most
    # recent revision per account.
    df = df[df["event_time"] <= T]
    df = df.sort_values(["account_id", "event_time"])
    latest = df.groupby("account_id", as_index=False).last()

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def score_row(r):
        z = W_F1 * r["f1"] + W_F2 * r["f2"] + W_F3 * r["f3"] + BIAS
        return round(sigmoid(z), 6)

    latest["score"] = latest.apply(score_row, axis=1)
    out = latest[["account_id", "score"]].copy()
    out["account_id"] = out["account_id"].astype(str)
    out["score"] = out["score"].astype("float64")

    print("Scoring %d accounts as of T=%d" % (len(out), T))
    print(out.head(10).to_string())

    # Create the result feature group: online-enabled for low-latency lookup.
    scores_fg = fs.get_or_create_feature_group(
        name="scoresb6c522",
        version=1,
        description="Batch scores as of T=%d (sigmoid logistic model)" % T,
        primary_key=["account_id"],
        online_enabled=True,
    )
    scores_fg.insert(out)
    print("Inserted %d rows into scoresb6c522 v1 (online_enabled)" % len(out))


if __name__ == "__main__":
    main()
