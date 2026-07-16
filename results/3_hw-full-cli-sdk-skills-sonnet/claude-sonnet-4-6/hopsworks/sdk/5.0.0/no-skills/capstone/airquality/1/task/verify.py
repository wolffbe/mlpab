"""Verify all deliverables are on the platform."""
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

print("=== Verifying deliverables ===\n")

# 1. Feature group
fg = fs.get_feature_group("airq3c8c0c", version=1)
print(f"[OK] Feature group airq3c8c0c: {fg.name} v{fg.version}")

# 2. Training dataset (via feature view)
fv = fs.get_feature_view("airqfv3c8c0c", version=1)
print(f"[OK] Feature view airqfv3c8c0c: {fv.name} v{fv.version}")

# 3. Model
model = mr.get_model("airqmodel3c8c0c", version=1)
print(f"[OK] Model airqmodel3c8c0c: {model.name} v{model.version}")
print(f"     Metrics: {model.training_metrics}")

# 4. Predictions feature group
pred_fg = fs.get_feature_group("airqpred3c8c0c", version=1)
print(f"[OK] Predictions FG airqpred3c8c0c: {pred_fg.name} v{pred_fg.version}")
print(f"     Online enabled: {pred_fg.online_enabled}")

pred_data = pred_fg.read()
print(f"     Rows: {len(pred_data)}")
print(f"     Columns: {list(pred_data.columns)}")
print(f"     pm25_pred stats: min={pred_data['pm25_pred'].min():.2f}, max={pred_data['pm25_pred'].max():.2f}, mean={pred_data['pm25_pred'].mean():.2f}")
print(f"     Sample predictions:")
print(pred_data[["date", "pm25_pred"]].head(10).to_string())
print(f"\nAll {len(pred_data)} predictions stored. Pipeline complete!")
