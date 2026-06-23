#!/usr/bin/env python3
"""Verify all pipeline deliverables on the platform."""
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

print("\n=== FEATURE GROUP: cctxn89f322 ===")
fg = fs.get_feature_group(name="cctxn89f322", version=1)
print(f"  Name: {fg.name}, v{fg.version}")
print(f"  Features: {[f.name for f in fg.features]}")
print(f"  Primary key: {fg.primary_key}")
print(f"  Event time: {fg.event_time}")

print("\n=== FEATURE VIEW: cctd89f322 ===")
fv = fs.get_feature_view(name="cctd89f322", version=1)
print(f"  Name: {fv.name}, v{fv.version}")
if fv:
    print(f"  Labels: {fv.labels}")

print("\n=== MODEL: ccmodel89f322 ===")
try:
    model = mr.get_best_model("ccmodel89f322", metric="roc_auc", direction="max")
    print(f"  Name: {model.name}, v{model.version}")
    print(f"  Metrics: {model.training_metrics}")
except Exception as e:
    print(f"  Error: {e}")
    try:
        model = mr.get_model("ccmodel89f322", version=1)
        print(f"  Name: {model.name}, v{model.version}")
        print(f"  Metrics: {model.training_metrics}")
    except Exception as e2:
        print(f"  Error v2: {e2}")

print("\n=== PREDICTIONS: ccpred89f322 ===")
pred_fg = fs.get_feature_group(name="ccpred89f322", version=1)
print(f"  Name: {pred_fg.name}, v{pred_fg.version}")
print(f"  Features: {[f.name for f in pred_fg.features]}")
print(f"  Online enabled: {pred_fg.online_enabled}")
print(f"  Primary key: {pred_fg.primary_key}")

# Sample some predictions
print("\n  Reading sample predictions …")
pred_df = pred_fg.read()
print(f"  Total rows: {len(pred_df)}")
print(f"  Columns: {list(pred_df.columns)}")
print(f"  fraud_probability range: [{pred_df['fraud_probability'].min():.4f}, {pred_df['fraud_probability'].max():.4f}]")
print(f"  fraud_probability mean: {pred_df['fraud_probability'].mean():.4f}")
print(f"  Predicted fraud rate (>0.5): {(pred_df['fraud_probability'] > 0.5).mean():.2%}")
print("\nAll deliverables verified!")
