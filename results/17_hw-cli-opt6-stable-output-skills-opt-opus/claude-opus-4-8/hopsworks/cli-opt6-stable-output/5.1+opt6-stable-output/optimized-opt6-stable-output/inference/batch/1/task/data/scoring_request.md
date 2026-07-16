# Batch scoring request

Score EVERY account AS OF **T = 1773223200000** (epoch milliseconds; 2026-03-11T10:00:00Z).

For each account, use the feature values that were VALID AT time T — the most recent revision in data/feature_history.csv with `event_time` (epoch milliseconds) at or before T. Revisions after T must not influence any score.

The model (data/model.json) is a logistic scorer:
    score = sigmoid(w_f1*f1 + w_f2*f2 + w_f3*f3 + bias)
rounded to 6 decimal places.
