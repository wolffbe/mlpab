"""MLflow model wrapper for the scorer."""
import json
import math

# Model weights (fixed at training time — do not change)
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


def _load_pyfunc(data_path):
    """Load the model."""
    return None


def predict(context, model_input):
    """Predict function for MLflow."""
    # Handle different input formats
    result = model_input
    if isinstance(model_input, dict):
        if "inputs" in model_input:
            result = model_input["inputs"]
        elif "data" in model_input:
            result = model_input["data"]
        elif "text" in model_input:
            result = model_input["text"]
    if isinstance(result, list):
        result = result[0]
    if isinstance(result, str):
        return score(result)
    return {"score": 0.0}
