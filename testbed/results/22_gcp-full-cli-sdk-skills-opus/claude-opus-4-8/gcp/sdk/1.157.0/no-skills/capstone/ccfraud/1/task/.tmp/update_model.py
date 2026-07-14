from google.cloud import aiplatform_v1
from google.protobuf import field_mask_pb2
import os, json
proj=os.environ['GCP_PROJECT']; loc=os.environ['GCP_LOCATION']; pref=os.environ['MLPAB_GCP_PREFIX']
ep=f"{loc}-aiplatform.googleapis.com"
client=aiplatform_v1.ModelServiceClient(client_options={"api_endpoint":ep}, transport="rest")

with open(".tmp/metrics.json") as f: metrics=json.load(f)
res=f"projects/{proj}/locations/{loc}/models/{pref}-ccmodel76ccb2"
new_dn=f"{pref}_ccmodel76ccb2"
desc="Fraud LOGISTIC_REG. Eval metrics: "+json.dumps(metrics)
# labels: only lowercase letters/numbers/dash/underscore; store roc_auc rounded
labels={"roc_auc_x1000": str(int(round(metrics.get('roc_auc',0)*1000))),
        "f1_x1000": str(int(round(metrics.get('f1_score',0)*1000))),
        "task":"fraud"}
m=aiplatform_v1.Model(name=res, display_name=new_dn, description=desc, labels=labels)
mask=field_mask_pb2.FieldMask(paths=["display_name","description","labels"])
out=client.update_model(model=m, update_mask=mask)
print("updated:", out.display_name)
print("desc:", out.description[:120])
print("labels:", dict(out.labels))
