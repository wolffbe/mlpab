"""Full FTI fraud pipeline — runs on the Hopsworks platform as a PYTHON job.

Stages:
  1. Feature engineering (model-independent) from raw CSVs in HopsFS.
  2. Write labelled history features -> feature group `cctxnd9a860`.
  3. Feature view + training dataset `cctdd9a860`.
  4. Train sklearn classifier, eval held-out ROC AUC, register `ccmodeld9a860`.
  5. Score score_transactions.csv -> online+offline feature group `ccpredd9a860`.
"""
import os
import numpy as np
import pandas as pd
import hopsworks

SUFFIX = "d9a860"
FG_FEAT = "cctxn" + SUFFIX
FV_NAME = "cctd" + SUFFIX          # feature view; its training dataset is the named TD artifact
MODEL_NAME = "ccmodel" + SUFFIX
FG_PRED = "ccpred" + SUFFIX

print("== login ==", flush=True)
project = hopsworks.login()
fs = project.get_feature_store()
dapi = project.get_dataset_api()

# ---------------------------------------------------------------- load raw data
dapi.download("Resources/ccfraud/transactions.csv", local_path="transactions.csv", overwrite=True)
dapi.download("Resources/ccfraud/score_transactions.csv", local_path="score_transactions.csv", overwrite=True)
hist = pd.read_csv("transactions.csv")
score = pd.read_csv("score_transactions.csv")
print("hist", hist.shape, "score", score.shape, flush=True)


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def engineer(df):
    """Per-card, point-in-time features using only PAST rows (no leakage)."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["datetime"], utc=True)
    df["ts"] = (df["dt"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype("int64")
    df = df.sort_values(["cc_num", "dt"]).reset_index(drop=True)
    g = df.groupby("cc_num", sort=False)

    prev_dt = g["dt"].shift(1)
    df["time_since_prev"] = (df["dt"] - prev_dt).dt.total_seconds()
    df["time_since_prev"] = df["time_since_prev"].fillna(1e7)

    prev_lat = g["lat"].shift(1)
    prev_long = g["long"].shift(1)
    df["geo_dist"] = haversine(df["lat"], df["long"], prev_lat, prev_long).fillna(0.0)
    df["speed"] = df["geo_dist"] / (df["time_since_prev"] / 3600.0 + 1e-3)

    cum_cnt = g.cumcount()
    cum_sum = g["amount"].cumsum() - df["amount"]
    card_mean = (cum_sum / cum_cnt.replace(0, np.nan))
    df["card_mean_amt"] = card_mean.fillna(df["amount"])
    df["amt_vs_mean"] = df["amount"] / (df["card_mean_amt"] + 1e-3)

    # rolling prior-txn counts within time windows (excludes self)
    c1h, c24h = [], []
    for _, grp in g:
        ts = grp["ts"].values
        idx = np.arange(len(ts))
        lo1 = np.searchsorted(ts, ts - 3600, "left")
        lo24 = np.searchsorted(ts, ts - 86400, "left")
        c1h.append(idx - lo1)
        c24h.append(idx - lo24)
    df["cnt_1h"] = np.concatenate(c1h) if c1h else np.array([])
    df["cnt_24h"] = np.concatenate(c24h) if c24h else np.array([])

    df["hour"] = df["dt"].dt.hour
    df["dayofweek"] = df["dt"].dt.dayofweek
    df["is_night"] = (df["hour"] < 6).astype("int64")
    df["log_amount"] = np.log1p(df["amount"])
    df["cat_code"] = df["category"].astype("category").cat.codes.astype("int64")
    df["merch_code"] = df["merchant"].astype("category").cat.codes.astype("int64")
    return df


FEATURES = ["amount", "log_amount", "hour", "dayofweek", "is_night",
            "time_since_prev", "geo_dist", "speed", "card_mean_amt",
            "amt_vs_mean", "cnt_1h", "cnt_24h", "cat_code", "merch_code"]

hist_f = engineer(hist)
hist_f["event_time"] = hist_f["ts"]
print("engineered hist", hist_f.shape, flush=True)

# ------------------------------------------------------------ feature group (history)
fg = fs.get_or_create_feature_group(
    name=FG_FEAT, version=1,
    description="Engineered credit-card fraud features (labelled history).",
    primary_key=["transaction_id"], event_time="event_time",
    online_enabled=False,
)
fg_cols = ["transaction_id", "cc_num", "event_time", "is_fraud"] + FEATURES
fg.insert(hist_f[fg_cols], write_options={"wait_for_job": True})
print("inserted FG", FG_FEAT, flush=True)

# ------------------------------------------------------------ feature view + TD
try:
    fs.get_feature_view(name=FV_NAME, version=1).delete()
except Exception as e:
    print("no existing fv", e, flush=True)
query = fg.select(FEATURES + ["is_fraud"])
fv = fs.create_feature_view(name=FV_NAME, version=1, query=query, labels=["is_fraud"])
print("created FV", FV_NAME, flush=True)
_ret = fv.create_train_test_split(
    test_size=0.2, description="Fraud training dataset cctd" + SUFFIX,
    write_options={"wait_for_job": True},
)
td_version = _ret[0] if isinstance(_ret, (tuple, list)) else _ret
print("created TD version", td_version, flush=True)

# ------------------------------------------------------------ train + evaluate
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=td_version)
X_train = X_train[FEATURES].astype("float64")
X_test = X_test[FEATURES].astype("float64")
y_train = np.asarray(y_train["is_fraud"]).astype(int)
y_test = np.asarray(y_test["is_fraud"]).astype(int)
print("train", X_train.shape, "test", X_test.shape,
      "fraud_rate", float(y_train.mean()), flush=True)

clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                     max_depth=6, l2_regularization=1.0,
                                     random_state=42)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
auc = float(roc_auc_score(y_test, proba))
acc = float(accuracy_score(y_test, (proba >= 0.5).astype(int)))
f1 = float(f1_score(y_test, (proba >= 0.5).astype(int)))
print("HELD-OUT ROC AUC =", auc, "acc", acc, "f1", f1, flush=True)

# refit on full data for best scoring quality
X_all = pd.concat([X_train, X_test]).reset_index(drop=True)
y_all = np.concatenate([y_train, y_test])
clf.fit(X_all, y_all)

# ------------------------------------------------------------ register model
import joblib
mr = project.get_model_registry()
mdir = "ccmodel_dir"
os.makedirs(mdir, exist_ok=True)
joblib.dump(clf, os.path.join(mdir, "model.pkl"))
input_example = X_all.head(1).to_dict(orient="records")[0]
metrics = {"roc_auc": auc, "accuracy": acc, "f1": f1}
model = mr.python.create_model(
    name=MODEL_NAME, metrics=metrics,
    description="Credit-card fraud classifier (HistGradientBoosting).",
    input_example=input_example, feature_view=fv,
)
model.save(mdir)
print("registered model", MODEL_NAME, "metrics", metrics, flush=True)

# ------------------------------------------------------------ score new transactions
# Compute score features with full per-card history context (history + score combined).
combo = pd.concat([
    hist.assign(_is_score=0),
    score.assign(is_fraud=-1, _is_score=1),
], ignore_index=True, sort=False)
combo_f = engineer(combo)
score_f = combo_f[combo_f["_is_score"] == 1].copy()
score_f = score_f.drop_duplicates(subset=["transaction_id"], keep="last")
print("score engineered", score_f.shape, flush=True)

Xs = score_f[FEATURES].astype("float64")
score_f["fraud_probability"] = clf.predict_proba(Xs)[:, 1].astype("float64")
preds = score_f[["transaction_id", "fraud_probability"]].copy()
preds["fraud_probability"] = preds["fraud_probability"].clip(0.0, 1.0)
print("preds", preds.shape, "range",
      float(preds.fraud_probability.min()), float(preds.fraud_probability.max()), flush=True)

pred_fg = fs.get_or_create_feature_group(
    name=FG_PRED, version=1,
    description="Fraud probability predictions for score_transactions.",
    primary_key=["transaction_id"],
    online_enabled=True,
)
pred_fg.insert(preds, write_options={"wait_for_job": True})
print("inserted predictions FG", FG_PRED, flush=True)
print("DONE pipeline. ROC_AUC=", auc, flush=True)
