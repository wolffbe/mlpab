import os, math
from google.cloud import bigquery, aiplatform_v1 as a
from google.protobuf.struct_pb2 import Value, Struct

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
prefix = os.environ['MLPAB_GCP_PREFIX']
ep = f"{loc}-aiplatform.googleapis.com"; copts = {"api_endpoint": ep}

# metrics from BQML held-out ML.EVALUATE
bq = bigquery.Client(project=proj); D = f"`{proj}.{ds}`"
ev = list(bq.query(f"SELECT * FROM ML.EVALUATE(MODEL {D}.airqmodelf3f1d8)").result())[0]
rmse = math.sqrt(ev.mean_squared_error)
metrics = dict(rootMeanSquaredError=rmse, meanAbsoluteError=ev.mean_absolute_error,
               rSquared=ev.r2_score)
print("metrics:", {k: round(v,4) for k,v in metrics.items()})

# locate the registered Vertex model
mc = a.ModelServiceClient(transport="rest", client_options=copts)
parent = f"projects/{proj}/locations/{loc}"
vid = f"{prefix}_airqmodelf3f1d8"
model = next(m for m in mc.list_models(parent=parent) if m.display_name == vid)
print("model:", model.name)

# also stamp metrics on model labels (queryable), then attach a ModelEvaluation
s = Struct(); s.update({k: float(v) for k,v in metrics.items()})
me = a.ModelEvaluation(
    display_name=f"{prefix}_airq_heldout_eval",
    metrics_schema_uri="gs://google-cloud-aiplatform/schema/modelevaluation/regression_metrics_1.0.0.yaml",
    metrics=Value(struct_value=s))
res = mc.import_model_evaluation(parent=model.name, model_evaluation=me)
print("model evaluation attached:", res.name)

# verify
evs = list(mc.list_model_evaluations(parent=model.name))
print("evaluations on model:", len(evs))
for e in evs:
    print("  ", e.name.split('/')[-1], dict(e.metrics))
