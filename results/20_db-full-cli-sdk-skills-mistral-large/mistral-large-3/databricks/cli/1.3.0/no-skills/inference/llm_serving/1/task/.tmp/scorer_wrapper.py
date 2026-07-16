import json
import sys
from scorer import score

def score_model(payload):
    """Wrapper function for model serving."""
    if isinstance(payload, dict) and 'data' in payload:
        text = payload['data']
    elif isinstance(payload, str):
        text = payload
    else:
        try:
            text = json.loads(payload)['data']
        except:
            text = str(payload)
    
    return score(text)