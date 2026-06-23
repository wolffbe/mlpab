import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("leakage_data", version=1)
df = fg.read()

feats = ["f1", "f2", "f3", "f4", "f5", "f6"]
label = df["label"].astype(float)

print("=== LEAKAGE_ANALYSIS_START ===")
print("rows:", len(df))

# Pearson correlation of each feature with the label
corrs = {}
for f in feats:
    c = df[f].astype(float).corr(label)
    corrs[f] = c
    print(f"corr_with_label {f} = {c:.6f}")

# Per-class means and standard deviations to gauge class separation
print("--- per-class stats ---")
sep = {}
g0 = df[label == 0]
g1 = df[label == 1]
for f in feats:
    m0, m1 = g0[f].mean(), g1[f].mean()
    s0, s1 = g0[f].std(), g1[f].std()
    pooled = ((s0 ** 2 + s1 ** 2) / 2.0) ** 0.5
    cohen_d = abs(m1 - m0) / pooled if pooled > 0 else float("inf")
    sep[f] = cohen_d
    print(f"{f}: mean0={m0:.4f} mean1={m1:.4f} std0={s0:.4f} std1={s1:.4f} cohen_d={cohen_d:.4f}")

best_corr = max(corrs, key=lambda k: abs(corrs[k]))
best_sep = max(sep, key=lambda k: sep[k])
print("MAX_ABS_CORR_FEATURE:", best_corr, abs(corrs[best_corr]))
print("MAX_SEPARATION_FEATURE:", best_sep, sep[best_sep])
print("=== LEAKAGE_ANALYSIS_END ===")
