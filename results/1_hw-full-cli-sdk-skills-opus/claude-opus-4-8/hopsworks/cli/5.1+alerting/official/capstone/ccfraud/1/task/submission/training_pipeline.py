"""T stage: assemble training dataset cctdfe5424 from the feature group, train a
fraud classifier, and register it as ccmodelfe5424 with evaluation metrics."""
import os
import sys
import json
import joblib
import numpy as np
import hopsworks
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)

project = hopsworks.login()

ds = project.get_dataset_api()
ds.download("Resources/ccdata/ccfraud_features.py", local_path=".", overwrite=True)
sys.path.insert(0, ".")
import ccfraud_features as fe

fs = project.get_feature_store()
fg = fs.get_feature_group("cctxnfe5424", version=1)

# Feature view = the assembled training-data definition (features + label),
# excluding identifiers and event time so nothing leaky reaches the model.
query = fg.select(fe.FEATURES + ["is_fraud"])
fv = fs.get_or_create_feature_view(
    name="cctdfe5424",
    version=1,
    query=query,
    labels=["is_fraud"],
    description="Training dataset for the credit-card fraud classifier",
)

# Materialize a versioned, point-in-time-correct train/test split (persisted TD).
td_version, td_job = fv.create_train_test_split(
    test_size=0.2, write_options={"wait_for_job": True}
)
X_train, X_test, y_train, y_test = fv.get_train_test_split(
    training_dataset_version=td_version
)

# Lock feature order.
X_train = X_train[fe.FEATURES]
X_test = X_test[fe.FEATURES]
y_train = np.asarray(y_train["is_fraud"]).astype(int)
y_test = np.asarray(y_test["is_fraud"]).astype(int)
print("Train:", X_train.shape, "Test:", X_test.shape,
      "train fraud rate:", round(float(y_train.mean()), 4))

model = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.08, max_depth=6,
    l2_regularization=1.0, random_state=42,
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
metrics = {
    "roc_auc": float(roc_auc_score(y_test, proba)),
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
}
print("Metrics:", json.dumps(metrics, indent=2))

model_dir = "ccmodelfe5424_dir"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, f"{model_dir}/model.pkl")
json.dump(fe.FEATURES, open(f"{model_dir}/feature_names.json", "w"))

mr = project.get_model_registry()
hw_model = mr.python.create_model(
    name="ccmodelfe5424",
    metrics=metrics,
    description="Credit-card fraud classifier (HistGradientBoosting) trained on cctdfe5424",
    input_example=X_train.head(1),
    feature_view=fv,
    training_dataset_version=td_version,
)
hw_model.save(model_dir)
print("Registered ccmodelfe5424 version", hw_model.version, "td_version", td_version)
