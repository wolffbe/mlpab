import math

# model weights (fixed at training time — do not change)
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


class ScorerModel:
    def __init__(self):
        pass
    
    def load_context(self, context):
        pass
    
    def predict(self, context, model_input):
        # Handle different input formats
        if isinstance(model_input, dict):
            if 'data' in model_input:
                text = model_input['data']
            elif 'inputs' in model_input:
                text = model_input['inputs']
            elif 'text' in model_input:
                text = model_input['text']
            else:
                text = str(model_input)
        else:
            text = str(model_input)
        
        result = score(text)
        return result


def _load_pyfunc(data_path):
    return ScorerModel()