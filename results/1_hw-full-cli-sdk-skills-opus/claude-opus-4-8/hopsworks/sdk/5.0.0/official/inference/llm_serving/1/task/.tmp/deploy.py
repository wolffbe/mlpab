import hopsworks
from hsml.resources import PredictorResources, Resources
from hsml.scaling_config import PredictorScalingConfig, ScaleMetric

DEPLOY_NAME = "scorerd0462a"
MODEL_NAME = "scorerd0462a"

project = hopsworks.login()
mr = project.get_model_registry()
ms = project.get_model_serving()

# Clean up any stale deployment with the same name.
try:
    existing = ms.get_deployment(DEPLOY_NAME)
    if existing is not None:
        print("Deleting stale deployment", DEPLOY_NAME)
        try:
            existing.stop(await_stopped=120)
        except Exception as e:
            print("stop:", e)
        existing.delete()
except Exception as e:
    print("no existing deployment:", e)

# Register the python model (scorer.py is saved as a model file).
try:
    model = mr.get_model(MODEL_NAME, version=1)
    print("Model already registered:", model.name, model.version)
except Exception:
    model = mr.python.create_model(
        name=MODEL_NAME,
        version=1,
        description="Deterministic trigram language-model scorer",
    )
    model.save(".tmp/model_dir")
    print("Registered model:", model.name, model.version)

# Upload predictor.py next to the model files.
script_dir = f"/Projects/{project.name}/Models/{model.name}/{model.version}/Files"
project.get_dataset_api().upload(".tmp/predictor.py", script_dir, overwrite=True)
print("Uploaded predictor to", script_dir)

deployment = model.deploy(
    name=DEPLOY_NAME,
    description="Real-time scorer endpoint",
    script_file=f"{script_dir}/predictor.py",
    resources=PredictorResources(
        requests=Resources(cores=1, memory=1024, gpus=0),
        limits=Resources(cores=2, memory=2048, gpus=0),
    ),
    scaling_configuration=PredictorScalingConfig(
        min_instances=1,
        max_instances=1,
        scale_metric=ScaleMetric.CONCURRENCY,
        target=70,
    ),
    environment="minimal-inference-pipeline",
)
print("Created deployment:", deployment.name)
deployment.start(await_running=600)
print("Running:", deployment.is_running())
