import os
from google.cloud import bigquery

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
client = bigquery.Client(project=project)
tbl = f"`{project}.{dataset}.training_data`"

feats = [f"f{i}" for i in range(1, 7)]

# 1) Correlation of each feature with label
corr_sel = ", ".join([f"CORR({f}, label) AS corr_{f}" for f in feats])
q1 = f"SELECT {corr_sel} FROM {tbl}"
print("=== Pearson correlation feature vs label ===")
for row in client.query(q1).result():
    for f in feats:
        print(f"  {f}: {row[f'corr_{f}']:+.4f}")

# 2) Class-conditional means/std to see separation (how many std apart classes are)
print("\n=== Class-conditional stats (mean0, mean1, std_pooled, separation) ===")
for f in feats:
    q = f"""
    SELECT
      AVG(IF(label=0,{f},NULL)) m0,
      AVG(IF(label=1,{f},NULL)) m1,
      STDDEV(IF(label=0,{f},NULL)) s0,
      STDDEV(IF(label=1,{f},NULL)) s1
    FROM {tbl}
    """
    r = list(client.query(q).result())[0]
    m0, m1, s0, s1 = r.m0, r.m1, r.s0, r.s1
    pooled = ((s0**2 + s1**2)/2) ** 0.5
    sep = abs(m1 - m0) / pooled if pooled else float('inf')
    print(f"  {f}: m0={m0:+.3f} m1={m1:+.3f} s0={s0:.3f} s1={s1:.3f} sep={sep:.3f}")

# 3) Best single-threshold classification accuracy per feature (AUC-like perfect separator check)
# Compute how well a threshold at the midpoint of class means separates
print("\n=== Single-feature threshold accuracy (mid-mean rule) ===")
for f in feats:
    q = f"""
    WITH stats AS (
      SELECT (AVG(IF(label=0,{f},NULL)) + AVG(IF(label=1,{f},NULL)))/2 AS thr,
             SIGN(AVG(IF(label=1,{f},NULL)) - AVG(IF(label=0,{f},NULL))) AS dir
      FROM {tbl}
    )
    SELECT AVG(CASE WHEN (({f} >= thr AND dir>=0) OR ({f} < thr AND dir<0)) = (label=1) THEN 1 ELSE 0 END) acc
    FROM {tbl}, stats
    """
    r = list(client.query(q).result())[0]
    print(f"  {f}: acc={r.acc:.4f}")
