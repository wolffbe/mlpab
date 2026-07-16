import warnings
warnings.filterwarnings("ignore")
import hopsworks, pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()
fv = fs.get_feature_view("churntrainingc8f821", version=1)

X, y = fv.get_training_data(training_dataset_version=1, dataframe_type="pandas")
df = pd.concat([X, y], axis=1)
print("Columns:", sorted(df.columns.tolist()))
print("Row count:", len(df))
required = {"account_id","label_time","amount","balance","credit_score",
            "tier","sessions_7d","health_score","churned"}
print("Has exactly required columns:", set(df.columns) == required)
print("Null counts:\n", df.isnull().sum())

# ---- Spot-check PIT correctness against raw CSVs ----
labels = pd.read_csv("data/labels.csv")
tx = pd.concat([pd.read_csv("data/transactions.csv"),
                pd.read_csv("data/transactions_late.csv")], ignore_index=True)
prof = pd.read_csv("data/profiles.csv")

def latest(src, acct, t, col):
    s = src[(src.account_id == acct) & (src.event_time <= t)]
    if s.empty:
        return None
    return s.sort_values("event_time").iloc[-1][col]

ok = True
for acct in labels.account_id.sample(8, random_state=1):
    t = int(labels[labels.account_id == acct].label_time.iloc[0])
    row = df[df.account_id == acct].iloc[0]
    exp_amt = latest(tx, acct, t, "amount")
    exp_cs = latest(prof, acct, t, "credit_score")
    amt_ok = (exp_amt is None and pd.isna(row.amount)) or (exp_amt is not None and abs(float(row.amount) - float(exp_amt)) < 1e-6)
    cs_ok = (exp_cs is None and pd.isna(row.credit_score)) or (exp_cs is not None and int(row.credit_score) == int(exp_cs))
    print(f"{acct} t={t} amount got={row.amount} exp={exp_amt} {amt_ok} | credit got={row.credit_score} exp={exp_cs} {cs_ok}")
    ok = ok and amt_ok and cs_ok
print("PIT spot-check passed:", ok)
