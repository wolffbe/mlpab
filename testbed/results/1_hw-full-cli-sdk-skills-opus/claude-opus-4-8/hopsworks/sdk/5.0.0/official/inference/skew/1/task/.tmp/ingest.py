import pandas as pd
import hopsworks
from hsfs.feature import Feature

proj = hopsworks.login()
fs = proj.get_feature_store()

train = pd.read_csv("data/training_sample.csv")
serve = pd.read_csv("data/serving_log.csv")
print("train shape", train.shape, "serve shape", serve.shape)

feats = lambda: [
    Feature("entity_id", "string", description="Entity identifier"),
    Feature("f1", "double", description="Feature 1"),
    Feature("f2", "double", description="Feature 2"),
    Feature("f3", "double", description="Feature 3"),
    Feature("f4", "double", description="Feature 4"),
    Feature("f5", "double", description="Feature 5"),
]

train_fg = fs.get_or_create_feature_group(
    name="skew_training", version=1,
    description="Training-path feature matrix",
    primary_key=["entity_id"], features=feats(),
    online_enabled=False, statistics_config=False,
)
train_fg.insert(train, wait=True)
print("training inserted")

serve_fg = fs.get_or_create_feature_group(
    name="skew_serving", version=1,
    description="Serving-path logged feature vectors",
    primary_key=["entity_id"], features=feats(),
    online_enabled=False, statistics_config=False,
)
serve_fg.insert(serve, wait=True)
print("serving inserted")
