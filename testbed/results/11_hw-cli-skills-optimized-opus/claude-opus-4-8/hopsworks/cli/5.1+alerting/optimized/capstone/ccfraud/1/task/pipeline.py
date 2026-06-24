"""Full FTI fraud pipeline — runs ON the Hopsworks platform as a job."""
import os
import numpy as np
import pandas as pd
import hopsworks

print(">>> logging in", flush=True)
project = hopsworks.login()
fs = project.get_feature_store()
dsapi = project.get_dataset_api()

# ---------------------------------------------------------------- load data
def load(remote):
    local = dsapi.download("Resources/ccdata/" + remote, overwrite=True)
    return pd.read_csv(local)

train = load("transactions.csv")
score = load("score_transactions.csv")
print(">>> train", train.shape, "score", score.shape, flush=True)

for df in (train, score):
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)

# --------------------------------------------------- card profiles (no leak)
prof = train.groupby("cc_num").agg(
    card_mean_lat=("lat", "mean"),
    card_mean_long=("long", "mean"),
    card_mean_amt=("amount", "mean"),
    card_std_amt=("amount", "std"),
    card_txn_count=("amount", "count"),
).reset_index()
g_lat, g_long = train["lat"].mean(), train["long"].mean()
g_amt, g_std = train["amount"].mean(), train["amount"].std()

def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

# velocity computed over union so score rows see prior history
train["_src"] = "train"
score["_src"] = "score"
allx = pd.concat([train, score], ignore_index=True, sort=False)
allx = allx.sort_values(["cc_num", "datetime"]).reset_index(drop=True)
allx["secs_since_prev"] = (
    allx.groupby("cc_num")["datetime"].diff().dt.total_seconds()
)
# count of same-card txns in the prior 1 hour
def cnt_1h(grp):
    t = grp.values.astype("datetime64[s]").astype("int64")
    out = np.zeros(len(t), dtype="float64")
    j = 0
    for i in range(len(t)):
        while t[i] - t[j] > 3600:
            j += 1
        out[i] = i - j
    return pd.Series(out, index=grp.index)
allx["txn_count_1h"] = allx.groupby("cc_num")["datetime"].transform(
    lambda s: cnt_1h(s)
)

def build(df):
    m = df.merge(prof, on="cc_num", how="left")
    m["card_mean_lat"] = m["card_mean_lat"].fillna(g_lat)
    m["card_mean_long"] = m["card_mean_long"].fillna(g_long)
    m["card_mean_amt"] = m["card_mean_amt"].fillna(g_amt)
    m["card_std_amt"] = m["card_std_amt"].fillna(g_std).replace(0, g_std)
    m["card_txn_count"] = m["card_txn_count"].fillna(0)
    m["log_amount"] = np.log1p(m["amount"])
    m["hour"] = m["datetime"].dt.hour
    m["dow"] = m["datetime"].dt.dayofweek
    m["geo_dist"] = haversine(m["lat"], m["long"], m["card_mean_lat"], m["card_mean_long"])
    m["amount_to_mean"] = m["amount"] / m["card_mean_amt"]
    m["amount_z"] = (m["amount"] - m["card_mean_amt"]) / m["card_std_amt"]
    return m

allx = build(allx)
train_f = allx[allx["_src"] == "train"].copy()
score_f = allx[allx["_src"] == "score"].copy()
print(">>> features built", flush=True)

FEATS_NUM = ["amount", "log_amount", "hour", "dow", "geo_dist",
             "amount_to_mean", "amount_z", "secs_since_prev", "txn_count_1h",
             "card_txn_count"]
FEAT_CAT = ["category"]
ALL_FEATS = FEATS_NUM + FEAT_CAT

# ----------------------------------------------------------- feature group
fg_cols = ["transaction_id", "cc_num", "datetime"] + ALL_FEATS + ["is_fraud"]
fg_df = train_f[fg_cols].copy()
fg_df["is_fraud"] = fg_df["is_fraud"].astype("int64")

fg = fs.get_or_create_feature_group(
    name="cctxn18c7ed", version=1,
    description="Engineered credit-card fraud features",
    primary_key=["transaction_id"], event_time="datetime",
    online_enabled=False,
)
fg.insert(fg_df, write_options={"wait_for_job": True})
print(">>> cctxn18c7ed inserted", fg_df.shape, flush=True)

# ------------------------------------------------- feature view + train data
try:
    old = fs.get_feature_view(name="cctd18c7ed", version=1)
    old.delete()
except Exception as e:
    print("no existing fv:", e, flush=True)

query = fg.select(ALL_FEATS + ["is_fraud"])
fv = fs.create_feature_view(
    name="cctd18c7ed", version=1,
    description="Fraud training dataset feature view",
    query=query, labels=["is_fraud"],
)
print(">>> feature view cctd18c7ed created", flush=True)
td_version, td_job = fv.create_train_test_split(
    test_size=0.2, description="cctd18c7ed training dataset",
    write_options={"wait_for_job": True},
)
print(">>> training dataset version", td_version, flush=True)

# --------------------------------------------------------------- train model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

X = train_f[ALL_FEATS].copy()
y = train_f["is_fraud"].astype(int).values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), FEAT_CAT)],
    remainder="passthrough",
)
clf = Pipeline([
    ("pre", pre),
    ("gb", HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_depth=6,
        l2_regularization=1.0, random_state=42)),
])
clf.fit(Xtr, ytr)
proba_te = clf.predict_proba(Xte)[:, 1]
auc = float(roc_auc_score(yte, proba_te))
acc = float(accuracy_score(yte, (proba_te >= 0.5).astype(int)))
f1 = float(f1_score(yte, (proba_te >= 0.5).astype(int)))
print(f">>> HELD-OUT ROC AUC = {auc:.4f}  acc={acc:.4f}  f1={f1:.4f}", flush=True)

# refit on all labelled data for final scoring
clf.fit(X, y)

# ------------------------------------------------------------ register model
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

mr = project.get_model_registry()
mdir = "ccmodel_dir"
os.makedirs(mdir, exist_ok=True)
joblib.dump(clf, os.path.join(mdir, "model.pkl"))
input_ex = X.head(3)
mschema = ModelSchema(input_schema=Schema(X), output_schema=Schema(y))
model = mr.python.create_model(
    name="ccmodel18c7ed",
    metrics={"roc_auc": auc, "accuracy": acc, "f1": f1},
    description="Credit-card fraud classifier (HistGradientBoosting)",
    input_example=input_ex,
    model_schema=mschema,
)
model.save(mdir)
print(">>> model ccmodel18c7ed registered with roc_auc", auc, flush=True)

# ------------------------------------------------------------------- score
score_proba = clf.predict_proba(score_f[ALL_FEATS])[:, 1]
pred_df = pd.DataFrame({
    "transaction_id": score_f["transaction_id"].values,
    "fraud_probability": np.clip(score_proba, 0.0, 1.0).astype("float64"),
})
print(">>> pred rows", pred_df.shape, "range",
      float(pred_df.fraud_probability.min()), float(pred_df.fraud_probability.max()),
      flush=True)

pred_fg = fs.get_or_create_feature_group(
    name="ccpred18c7ed", version=1,
    description="Fraud probability predictions for scored transactions",
    primary_key=["transaction_id"],
    online_enabled=True,
)
pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print(">>> ccpred18c7ed inserted", pred_df.shape, flush=True)
print(">>> PIPELINE DONE  AUC=", auc, flush=True)
