import mlflow
import json
import math

# Scorer constants and functions
A = 2.653901
B = 1.890485
C = 1.091312
D = 1.69396

def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)

def score(text):
    """Log-likelihood of `text` under the trigram model."""
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}

# Custom PythonModel that handles serving endpoint input format
class ScorerModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        super().__init__()
    
    def load_context(self, context):
        pass
    
    def predict(self, context, model_input):
        """
        Handle serving endpoint input.
        Serving endpoint sends: {"inputs": ["text1", "text2", ...]}
        MLflow wraps this and passes it as model_input.
        """
        # Extract text(s) from various possible input formats
        if isinstance(model_input, dict):
            # Format 1: {"inputs": ["text1", "text2", ...]}
            if "inputs" in model_input:
                inputs = model_input["inputs"]
                if isinstance(inputs, list):
                    texts = inputs
                else:
                    texts = [inputs]
            # Format 2: {"instances": [{"text": "t1"}, ...]}
            elif "instances" in model_input:
                texts = [inst.get("text", "") for inst in model_input["instances"]]
            # Format 3: {"text": "single text"}
            elif "text" in model_input:
                texts = [model_input["text"]]
            else:
                texts = [str(model_input)]
        elif hasattr(model_input, 'columns'):
            # pandas DataFrame - get first column
            texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, list):
            # List of strings or list of dicts
            if model_input and isinstance(model_input[0], dict):
                texts = [item.get("text", "") for item in model_input]
            else:
                texts = model_input
        else:
            texts = [str(model_input)]
        
        # Score each text and return list of results
        return [score(text) for text in texts]

# Set MLflow tracking
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/benedict@hopsworks.ai/mlpabaf8386/scorer_experiment")

# Start run and log model
with mlflow.start_run() as run:
    model = ScorerModel()
    
    # Log the model with input example
    mlflow.pyfunc.log_model(
        artifact_path="scorer_model",
        python_model=model,
        input_example={"inputs": ["test text"]}
    )
    
    # Register in Unity Catalog
    model_name = "workspace.mlpabaf8386.scorer40bb09"
    mv = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/scorer_model",
        name=model_name
    )
    
    print(f"Registered model: {model_name}, version: {mv.version}")
