"""Predictor wrapper for the trigram scorer (v2)."""
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
        """Predict method for KServe compatibility.
        
        Args:
            inputs: Either a dict (single payload) or a list of dicts (batch).
        """
        if isinstance(inputs, list):
            # Batch mode: list of dicts
            return [score(item.get("text", "")) for item in inputs]
        else:
            # Single payload: dict
            payload = inputs.get("text", "")
            return score(payload)