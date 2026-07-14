"""Platform-side verification: online store lookup + training dataset readback."""
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

pred_fg = fs.get_feature_group("airqpred754fa9", version=1)
online_df = pred_fg.read(online=True)
print(f"ONLINE_ROWS={len(online_df)}")
print(online_df.sort_values("date").head(3).to_string())

fv = fs.get_feature_view("airqtd754fa9", version=1)
X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=1)
print(f"TD_TRAIN={len(X_train)} TD_TEST={len(X_test)}")

mr = project.get_model_registry()
m = mr.get_model("airqmodel754fa9", version=1)
print(f"MODEL_METRICS={m.training_metrics}")
print("VERIFY_OK")
