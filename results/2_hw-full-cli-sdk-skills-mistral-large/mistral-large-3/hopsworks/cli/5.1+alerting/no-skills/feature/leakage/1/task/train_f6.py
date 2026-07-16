import hopsworks
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score

project = hopsworks.login()
fs = project.get_feature_store()

fv = fs.get_feature_view('leakage_fv_f6', 1)
td = fv.get_training_data(1)

X = td[0][['f6']]
y = td[0]['label']

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f'Accuracy for f6: {accuracy}')

mr = project.get_model_registry()
model_dir = 'f6_model'
model.booster().save_model(f'{model_dir}/model.json')

input_example = X.iloc[:1].to_json()
model_meta = mr.python.create_model('f6_model', metrics={'accuracy': accuracy}, input_example=input_example)
model_meta.save(model_dir)