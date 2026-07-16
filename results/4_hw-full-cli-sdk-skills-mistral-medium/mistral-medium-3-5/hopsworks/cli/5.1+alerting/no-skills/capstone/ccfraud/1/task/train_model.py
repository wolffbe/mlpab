import hopsworks
from hopsworks_common.project import Project
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import os

# Connect to Hopsworks
hopsworks.login()
project = Project()
fs = project.get_feature_store()

# Get the feature group
fg = fs.get_feature_group("cctxnee3558", version=1)

# Read all data from the feature group
df = fg.read()

print(f"Training dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label column: is_fraud")

# Check if we have the label
if 'is_fraud' not in df.columns:
    print("ERROR: is_fraud column not found in training dataset")
    # Try to get it from the feature view
    df = fv.get_batch_data()
    print(f"After get_batch_data: {df.columns.tolist()}")

# Separate features and labels
# Drop transaction_id and convert categorical columns
X = df.drop(columns=['is_fraud', 'transaction_id'])
y = df['is_fraud']

# Encode categorical columns
categorical_cols = ['merchant', 'category']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
model_dir = "/hopsfs/Resources/model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "ccmodelee3558.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Save metrics
metrics = {
    'roc_auc': roc_auc,
    'classification_report': classification_report(y_test, y_pred, output_dict=True)
}
metrics_path = os.path.join(model_dir, "ccmodelee3558_metrics.json")
import json
with open(metrics_path, 'w') as f:
    json.dump(metrics, f)
print(f"Metrics saved to {metrics_path}")

print("Training complete!")
print(f"Model saved to {model_path}")
print(f"Metrics: ROC AUC = {roc_auc}")
