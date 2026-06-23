"""Runs ON the Hopsworks platform.
Reads training data from feature view cctd89f322, trains GBM, registers ccmodel89f322.
Usage: python train_job.py <td_version>
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import hopsworks
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

td_version = int(sys.argv[1]) if len(sys.argv) > 1 else 1

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

fv = fs.get_feature_view("cctd89f322", version=1)
print(f"Reading training data version {td_version}...")
X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=td_version)

# Keep only numeric feature columns
SKIP_COLS = ["transaction_id", "cc_num", "is_fraud"]
num_cols = [c for c in X_train.columns if c not in SKIP_COLS and X_train[c].dtype in (np.float64, np.float32, np.int64, np.int32, int, float)]
print(f"Feature columns ({len(num_cols)}): {num_cols}")

X_tr = X_train[num_cols].fillna(0).values
X_te = X_test[num_cols].fillna(0).values
y_tr = y_train.values.ravel() if hasattr(y_train, 'values') else np.array(y_train).ravel()
y_te = y_test.values.ravel() if hasattr(y_test, 'values') else np.array(y_test).ravel()

print(f"Train: {X_tr.shape}, Test: {X_te.shape}")

model = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, random_state=42,
)
model.fit(X_tr, y_tr)

y_prob = model.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, y_prob)
print(f"Test ROC AUC: {auc:.4f}")

os.makedirs("ccmodel89f322", exist_ok=True)
with open("ccmodel89f322/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("ccmodel89f322/feature_cols.txt", "w") as f:
    f.write("\n".join(num_cols))

hw_model = mr.sklearn.create_model(
    name="ccmodel89f322",
    metrics={"roc_auc": float(auc)},
    description="CC fraud GBM classifier",
    input_example=pd.DataFrame(X_tr[:3], columns=num_cols),
)
hw_model.save("ccmodel89f322")
print(f"Model registered: ccmodel89f322 v1, AUC={auc:.4f}")
