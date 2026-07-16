"""A tiny deterministic pure-python "language model".

Character-trigram log-likelihood scorer: every trigram of the input text
contributes a weight derived from fixed constants; the score is the sum,
rounded to 6 decimal places. No dependencies beyond the standard library;
fully deterministic — the same text always yields the same score.
"""
import json
import math

# model weights (fixed at training time — do not change)
A = 0.830259
B = 1.593097
C = 0.500489
D = 1.756563


def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)


def score(text):
    """Log-likelihood of `text` under the trigram model."""
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}


class Predict:
    def __init__(self):
        pass

    def predict(self, inputs):
        """Predict method for Hopsworks serving.
        
        Expects the Hopsworks serving format: [{"text": "..."}]
        """
        import json
        import traceback
        
        try:
            # Log the input for debugging
            print(f"Received input: {json.dumps(inputs)}")
            
            # Extract text from the first instance
            if isinstance(inputs, list) and len(inputs) > 0:
                if isinstance(inputs[0], dict) and "text" in inputs[0]:
                    text = inputs[0]["text"]
                elif isinstance(inputs[0], str):
                    text = inputs[0]
                else:
                    raise ValueError("Instance must be a dict with a 'text' key or a string.")
            else:
                raise ValueError("Input must be a non-empty list of instances.")
            
            return score(text)
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            print(traceback.format_exc())
            raise