"""Full FTI fraud pipeline — runs ON the Hopsworks platform as a job."""
import os
import numpy as np
import pandas as pd
import hopsworks

print(">>> logging in", flush=True)
project = hopsworks.login()
fs = project.get_feature_store()
ds = project.get_dataset_api()

# ---------------------------------------------------------------- load inputs
for f in ("transactions.csv", "score_transactions.csv"):
    try:
        ds.download(f"Resources/ccfraud/{f}", local_path=f, overwrite=True)
    except Exception as e:
        print("download note:", e, flush=True)

hist = pd.read_csv("transactions.csv")
score = pd.read_csv("score_transactions.csv")
print(">>> hist", hist.shape, "score", score.shape, flush=True)

# ----------------------------------------------------- combined feature build
hist = hist.copy()
score = score.copy()
hist["is_fraud"] = hist["is_fraud"].astype(int)
score["is_fraud"] = np.nan
hist["__src"] = "hist"
score["__src"] = "score"
df = pd.concat([hist, score], ignore_index=True, sort=False)

df["dt"] = pd.to_datetime(df["datetime"], utc=True)
df["event_time"] = df["dt"].dt.tz_localize(None)
df = df.sort_values(["cc_num", "dt"]).reset_index(drop=True)

g = df.groupby("cc_num", sort=False)
df["log1p_amount"] = np.log1p(df["amount"])
df["hour"] = df["dt"].dt.hour
df["is_night"] = (df["hour"] < 6).astype(int)
df["dow"] = df["dt"].dt.dayofweek

df["n_prev"] = g.cumcount()
df["secs_since_prev"] = g["dt"].diff().dt.total_seconds()

df["amt_mean_prev"] = g["amount"].transform(lambda s: s.expanding().mean().shift(1))
df["amt_std_prev"] = g["amount"].transform(lambda s: s.expanding().std().shift(1))
df["prev_lat"] = g["lat"].shift(1)
df["prev_long"] = g["long"].shift(1)
df["home_lat"] = g["lat"].transform(lambda s: s.expanding().mean().shift(1))
df["home_long"] = g["long"].transform(lambda s: s.expanding().mean().shift(1))


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


df["dist_from_prev_km"] = haversine(df["lat"], df["long"], df["prev_lat"], df["prev_long"])
df["dist_from_home_km"] = haversine(df["lat"], df["long"], df["home_lat"], df["home_long"])

gmean = float(df["amount"].mean())
df["amt_mean_prev"] = df["amt_mean_prev"].fillna(gmean)
df["amt_std_prev"] = df["amt_std_prev"].fillna(0.0)
df["secs_since_prev"] = df["secs_since_prev"].fillna(1e7)
df["dist_from_prev_km"] = df["dist_from_prev_km"].fillna(0.0)
df["dist_from_home_km"] = df["dist_from_home_km"].fillna(0.0)
df["amt_z"] = (df["amount"] - df["amt_mean_prev"]) / (df["amt_std_prev"] + 1.0)
df["amt_ratio"] = df["amount"] / (df["amt_mean_prev"] + 1.0)

# category one-hot (fixed set from combined data)
cats = sorted(df["category"].dropna().astype(str).unique().tolist())
cat_cols = []
for c in cats:
    col = "cat_" + "".join(ch if ch.isalnum() else "_" for ch in c.lower())
    df[col] = (df["category"].astype(str) == c).astype(int)
    cat_cols.append(col)

FEATURES = ["amount", "log1p_amount", "hour", "is_night", "dow", "n_prev",
            "secs_since_prev", "amt_mean_prev", "amt_std_prev", "amt_z",
            "amt_ratio", "dist_from_prev_km", "dist_from_home_km"] + cat_cols
print(">>> n_features", len(FEATURES), flush=True)

base = ["transaction_id", "cc_num", "event_time"]
hist_feat = df[df["__src"] == "hist"][base + FEATURES + ["is_fraud"]].copy()
hist_feat["is_fraud"] = hist_feat["is_fraud"].astype(int)
score_feat = df[df["__src"] == "score"][base + FEATURES].copy()
print(">>> hist_feat", hist_feat.shape, "score_feat", score_feat.shape, flush=True)

# ----------------------------------------------------------- feature group FG
fg = fs.get_or_create_feature_group(
    name="cctxnd909a2", version=1,
    description="Engineered fraud features for labelled card transactions",
    primary_key=["transaction_id"], event_time="event_time",
    online_enabled=False,
)
fg.insert(hist_feat, write_options={"wait_for_job": True})
print(">>> inserted FG cctxnd909a2", flush=True)

# ------------------------------------------------------------- feature view FV
try:
    query = fg.select(FEATURES + ["is_fraud"])
    fv = fs.get_or_create_feature_view(
        name="cctdd909a2", version=1, query=query, labels=["is_fraud"],
        description="Fraud training feature view",
    )
    print(">>> feature view cctdd909a2 ready", flush=True)
except Exception as e:
    print("!! fv create error:", repr(e), flush=True)
    fv = fs.get_feature_view("cctdd909a2", version=1)

# ---------------------------------------------------- training dataset (TD)
Xtr = Xte = ytr = yte = None
try:
    Xtr, Xte, ytr, yte = fv.train_test_split(test_size=0.2)
    print(">>> TD materialized + read from FV; train", Xtr.shape, flush=True)
except Exception as e:
    print("!! fv.train_test_split failed, falling back to in-memory:", repr(e), flush=True)
    try:
        fv.create_train_test_split(test_size=0.2, data_format="csv",
                                   write_options={"wait_for_job": True})
        print(">>> TD materialized via create_train_test_split", flush=True)
    except Exception as e2:
        print("!! create_train_test_split also failed:", repr(e2), flush=True)

if Xtr is None:
    from sklearn.model_selection import train_test_split
    X = hist_feat[FEATURES]
    y = hist_feat["is_fraud"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y)

# align columns
Xtr = Xtr[FEATURES]
Xte = Xte[FEATURES]
ytr = np.asarray(ytr).ravel().astype(int)
yte = np.asarray(yte).ravel().astype(int)

# ----------------------------------------------------------------- train model
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                     max_depth=6, l2_regularization=1.0,
                                     random_state=42)
clf.fit(Xtr, ytr)
proba_te = clf.predict_proba(Xte)[:, 1]
auc = float(roc_auc_score(yte, proba_te))
pred_te = (proba_te >= 0.5).astype(int)
acc = float(accuracy_score(yte, pred_te))
f1 = float(f1_score(yte, pred_te))
print(f">>> HELD-OUT ROC AUC={auc:.4f} ACC={acc:.4f} F1={f1:.4f}", flush=True)

# ------------------------------------------------------------- register model
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

mdir = "ccmodel_dir"
os.makedirs(mdir, exist_ok=True)
joblib.dump({"model": clf, "features": FEATURES}, os.path.join(mdir, "model.pkl"))

mr = project.get_model_registry()
input_ex = Xtr.head(3)
model_schema = ModelSchema(
    input_schema=Schema(Xtr),
    output_schema=Schema(pd.DataFrame({"fraud_probability": [0.0]})),
)
model = mr.python.create_model(
    name="ccmodeld909a2",
    metrics={"roc_auc": auc, "accuracy": acc, "f1": f1},
    description="Credit-card fraud classifier (HistGradientBoosting)",
    input_example=input_ex,
    model_schema=model_schema,
)
model.save(mdir)
print(">>> registered model ccmodeld909a2 with metrics", flush=True)

# ----------------------------------------------------------- score + write FG
proba_score = clf.predict_proba(score_feat[FEATURES])[:, 1]
preds = pd.DataFrame({
    "transaction_id": score_feat["transaction_id"].values,
    "fraud_probability": np.clip(proba_score, 0.0, 1.0).astype(float),
})
preds["event_time"] = pd.Timestamp("2025-12-31")
print(">>> preds", preds.shape, "range",
      float(preds.fraud_probability.min()), float(preds.fraud_probability.max()),
      flush=True)

pred_fg = fs.get_or_create_feature_group(
    name="ccpredd909a2", version=1,
    description="Fraud probability predictions for scored transactions",
    primary_key=["transaction_id"], event_time="event_time",
    online_enabled=True,
)
pred_fg.insert(preds, write_options={"wait_for_job": True})
print(">>> inserted predictions FG ccpredd909a2 (online+offline)", flush=True)
print(">>> PIPELINE COMPLETE", flush=True)
