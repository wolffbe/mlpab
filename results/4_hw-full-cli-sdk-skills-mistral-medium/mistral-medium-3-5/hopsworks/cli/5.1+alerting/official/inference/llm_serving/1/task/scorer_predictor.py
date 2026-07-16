"""Predictor for scorer deployment."""
import json

# Import the score function from scorer module
# The scorer.py will be in the same directory as this script
import sys
import os

# The working directory will have scorer.py
sys.path.insert(0, os.getcwd())

# We need to import from the local scorer module
# But in deployment, the files are in the model directory
# Let's try to import directly
try:
    from scorer import score
except ImportError:
    # If scorer is not available, define it inline
    import math
    A = 2.91983
    B = 2.756296
    C = 2.470743
    D = -0.022418
    
    def _trigram_weight(tri):
        o0, o1, o2 = (ord(ch) for ch in tri)
        return math.sin(A * o0 + B * o1 + C * o2 + D)
    
    def score(text):
        """Log-likelihood of `text` under the trigram model."""
        ll = 0.0
        for i in range(len(text) - 2):
            ll += _trigram_weight(text[i:i + 3])
        return {"score": round(ll, 6)}


def predict(input_data):
    """Entry point for the deployment.
    
    input_data is the JSON body of the request.
    """
    # Parse input - could be dict with 'text' key or just the text
    if isinstance(input_data, dict):
        if 'text' in input_data:
            text = input_data['text']
        elif 'instances' in input_data:
            # Batch inference
            text = input_data['instances'][0] if input_data['instances'] else ""
        else:
            # Try to extract text from first value
            text = list(input_data.values())[0] if input_data else ""
    else:
        text = str(input_data)
    
    result = score(text)
    return result
