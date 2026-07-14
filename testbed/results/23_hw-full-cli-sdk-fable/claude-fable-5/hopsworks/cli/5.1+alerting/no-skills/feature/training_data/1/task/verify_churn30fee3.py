"""Platform job: verify training dataset churntraining30fee3 v1 (schema, rows, PIT correctness)."""
import pandas as pd
import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()

td_path = (
    "mlpabaa1fcd_Training_Datasets/churntraining30fee3_1_1/"
    "churntraining30fee3_1/part-00000.parquet"
)
local_td = dataset_api.download(td_path, overwrite=True)
td = pd.read_parquet(local_td)
print("SHAPE:", td.shape)
print("COLUMNS:", list(td.columns))
print("NULLS:", {k: int(v) for k, v in td.isna().sum().items() if v})

BASE = "Resources/churn30fee3"
def dl(name):
    return pd.read_csv(dataset_api.download(f"{BASE}/{name}", overwrite=True))

tx = pd.concat([dl("transactions.csv"), dl("transactions_late.csv")]).drop_duplicates()
prof = dl("profiles.csv").drop_duplicates()
act = dl("activity.csv").drop_duplicates()
health = dl("account_health.csv").drop_duplicates()
labels = dl("labels.csv")

def asof(labels, feat, cols):
    feat = feat.sort_values("event_time")
    lab = labels.sort_values("label_time")
    out = pd.merge_asof(
        lab, feat, left_on="label_time", right_on="event_time",
        by="account_id", direction="backward",
    )
    return out[["account_id", "label_time"] + cols]

exp = labels.copy()
for feat, cols in [(tx, ["amount", "balance"]), (prof, ["credit_score", "tier"]),
                   (act, ["sessions_7d"]), (health, ["health_score"])]:
    exp = exp.merge(asof(labels, feat, cols), on=["account_id", "label_time"], how="left")

exp = exp[["account_id", "label_time", "amount", "balance", "credit_score",
           "tier", "sessions_7d", "health_score", "churned"]]

m = td.merge(exp, on=["account_id", "label_time"], suffixes=("_td", "_exp"))
print("MERGED ROWS:", len(m), "(expected", len(labels), ")")
bad = 0
for c in ["amount", "balance", "credit_score", "tier", "sessions_7d", "health_score", "churned"]:
    a, b = m[c + "_td"], m[c + "_exp"]
    if m[c + "_td"].dtype == object:
        neq = ~((a == b) | (a.isna() & b.isna()))
    else:
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        neq = ~(((a - b).abs() < 1e-6) | (a.isna() & b.isna()))
    if neq.any():
        bad += int(neq.sum())
        print("MISMATCH", c, int(neq.sum()))
        print(m.loc[neq, ["account_id", "label_time", c + "_td", c + "_exp"]].head(5).to_string())
print("TOTAL MISMATCHED VALUES:", bad)
print(td.sort_values("account_id").head(8).to_string())
