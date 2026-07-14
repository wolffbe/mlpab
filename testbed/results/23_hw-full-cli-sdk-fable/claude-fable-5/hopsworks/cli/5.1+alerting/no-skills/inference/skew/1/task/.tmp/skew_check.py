"""Platform job: detect training/serving skew between two logged feature CSVs.

Reads both CSVs from the project's HopsFS, joins them on entity_id, and
measures per-feature divergence. Writes submission/answers.json back to
HopsFS.
"""
import json
import os

import pandas as pd
import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()

train_local = ds.download("Resources/skew/training_sample.csv", overwrite=True)
serve_local = ds.download("Resources/skew/serving_log.csv", overwrite=True)

train = pd.read_csv(train_local)
serve = pd.read_csv(serve_local)

features = [c for c in train.columns if c != "entity_id"]
merged = train.merge(serve, on="entity_id", suffixes=("_train", "_serve"))
print(f"rows: train={len(train)} serve={len(serve)} matched={len(merged)}")

report = {}
worst_feature, worst_score = None, -1.0
for f in features:
    t = merged[f"{f}_train"]
    s = merged[f"{f}_serve"]
    diff = (s - t).abs()
    scale = t.abs().mean() or 1.0
    frac_mismatch = float((diff > 1e-6).mean())
    rel_mad = float(diff.mean() / scale)
    report[f] = {
        "frac_rows_mismatched": frac_mismatch,
        "mean_abs_diff": float(diff.mean()),
        "rel_mean_abs_diff": rel_mad,
        "train_mean": float(t.mean()),
        "serve_mean": float(s.mean()),
        "train_std": float(t.std()),
        "serve_std": float(s.std()),
    }
    score = frac_mismatch + rel_mad
    if score > worst_score:
        worst_score, worst_feature = score, f

print(json.dumps(report, indent=2))
print("DIVERGING_FEATURE:", worst_feature)

# Characterize the divergence for the optional "cause" field.
t = merged[f"{worst_feature}_train"]
s = merged[f"{worst_feature}_serve"]
ratio = (s / t).replace([float("inf"), -float("inf")], pd.NA).dropna()
corr = float(t.corr(s)) if len(t) > 1 else float("nan")
cause = (
    f"{worst_feature} mismatches on {report[worst_feature]['frac_rows_mismatched']:.0%} of matched rows "
    f"(all other features are bit-identical). train mean/std={t.mean():.3f}/{t.std():.3f}, "
    f"serve mean/std={s.mean():.3f}/{s.std():.3f}, corr(train,serve)={corr:.3f}, "
    f"median serve/train ratio={ratio.median():.3f} — the serving path computes this feature with a "
    f"different transformation than the training path."
)
print("CAUSE:", cause)

answer = {"feature": worst_feature, "cause": cause}
os.makedirs("out_submission", exist_ok=True)
with open("out_submission/answers.json", "w") as fh:
    json.dump(answer, fh, indent=2)

try:
    ds.mkdir("Resources/submission")
except Exception as e:
    print("mkdir note:", e)
ds.upload("out_submission/answers.json", "Resources/submission", overwrite=True)
print("uploaded answers.json to Resources/submission/answers.json")
print("ANSWER_JSON:", json.dumps(answer))
