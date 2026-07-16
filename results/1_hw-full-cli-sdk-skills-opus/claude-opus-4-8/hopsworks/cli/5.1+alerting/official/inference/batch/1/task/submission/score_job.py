"""Batch scoring job — runs ON the Hopsworks platform.

Reads the raw feature-history feature group, selects each account's most recent
revision valid at or before T, applies the logistic model, and writes the
result to an online-enabled feature group `scores30c485` (v1).
"""
import math
import hopsworks

# Scoring request constants
T = 1773410400000  # as-of timestamp (epoch ms)
W_F1 = 1.1161
W_F2 = 0.6773
W_F3 = 0.155
BIAS = 0.2799


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()

    # 1. Read all raw revisions from the offline store.
    src = fs.get_feature_group("feature_history_raw", version=1)
    df = src.read()
    print(f"Read {len(df)} raw revisions")

    # 2. Point-in-time selection: most recent revision at or before T per account.
    valid = df[df["event_time"] <= T].copy()
    valid = valid.sort_values("event_time")
    latest = valid.groupby("account_id", as_index=False).last()
    print(f"{len(latest)} accounts have a revision valid at or before T")

    # 3. Score with the logistic model, rounded to 6 decimals.
    def score_row(r):
        z = W_F1 * r["f1"] + W_F2 * r["f2"] + W_F3 * r["f3"] + BIAS
        return round(1.0 / (1.0 + math.exp(-z)), 6)

    latest["score"] = latest.apply(score_row, axis=1)
    out = latest[["account_id", "score"]].copy()
    out["account_id"] = out["account_id"].astype(str)
    out["score"] = out["score"].astype(float)
    print(out.head(10).to_string())
    print(f"Scored {len(out)} accounts")

    # 4. Create the online-enabled target feature group and insert.
    scores_fg = fs.get_or_create_feature_group(
        name="scores30c485",
        version=1,
        description="Batch logistic scores as-of T per account",
        primary_key=["account_id"],
        online_enabled=True,
    )
    scores_fg.insert(out)
    print("Inserted scores into scores30c485 v1 (online enabled)")


if __name__ == "__main__":
    main()
