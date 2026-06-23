"""Training + batch-inference pipeline. Runs ON the Hopsworks platform as a job
(pandas-training-pipeline env). Trains a fraud classifier from feature view
cctdfe5424, registers it as ccmodelfe5424 with metrics, scores ccscorefe5424,
and writes fraud probabilities to online feature group ccpredfe5424.
"""
import os
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import hopsworks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

FEATURES = [
    "amount", "log_amount", "hour", "is_night",
    "time_since_prev_s", "dist_prev_km", "speed_kmh",
    "amt_z", "dist_home_km", "count_1h", "count_24h",
    "cat_fraud_rate",
]

proj = hopsworks.login()
fs = proj.get_feature_store()
fv = fs.get_feature_view(name="cctdfe5424", version=1)

# ---- training data (held-out split for honest evaluation)
X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2, seed=42)
X_train = X_train[FEATURES].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
X_test = X_test[FEATURES].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
y_train = y_train.values.ravel().astype(int)
y_test = y_test.values.ravel().astype(int)
print("train", X_train.shape, "test", X_test.shape,
      "pos_rate", float(y_train.mean()))

# ---- train
model = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    class_weight="balanced_subsample", n_jobs=-1, random_state=42,
)
model.fit(X_train, y_train)

# ---- evaluate
proba_test = model.predict_proba(X_test)[:, 1]
roc_auc = float(roc_auc_score(y_test, proba_test))
pr_auc = float(average_precision_score(y_test, proba_test))
f1 = float(f1_score(y_test, (proba_test >= 0.5).astype(int), zero_division=0))
metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1}
print("METRICS", metrics)

imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
print("IMPORTANCES", imp)

# ---- register model
model_dir = "ccmodelfe5424_dir"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
json.dump(FEATURES, open(os.path.join(model_dir, "feature_names.json"), "w"))

mr = proj.get_model_registry()
hw_model = mr.python.create_model(
    name="ccmodelfe5424",
    metrics=metrics,
    description="Credit-card fraud classifier (RandomForest) trained on cctdfe5424. "
                "Outputs fraud probability.",
    input_example=X_train.head(1),
    feature_view=fv,
    training_dataset_version=1,
)
hw_model.save(model_dir)
print("REGISTERED ccmodelfe5424")

# ---- batch inference on the scoring slice
score_fg = fs.get_feature_group("ccscorefe5424", version=1)
score_df = score_fg.read()
print("score rows", len(score_df))
Xs = score_df[FEATURES].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
fraud_probability = model.predict_proba(Xs)[:, 1]
fraud_probability = np.clip(fraud_probability, 0.0, 1.0)

pred_df = pd.DataFrame({
    "transaction_id": score_df["transaction_id"].values,
    "fraud_probability": fraud_probability.astype(float),
})
print("pred head\n", pred_df.head())
print("pred stats", float(pred_df.fraud_probability.min()),
      float(pred_df.fraud_probability.max()), len(pred_df))

# ---- predictions feature group (online + offline for low-latency lookup)
pred_fg = fs.get_or_create_feature_group(
    name="ccpredfe5424", version=1,
    description="Predicted fraud probability per scored transaction.",
    primary_key=["transaction_id"],
    online_enabled=True,
)
pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print("WROTE ccpredfe5424 rows", len(pred_df))
print("DONE_JOB roc_auc=%.4f" % roc_auc)
