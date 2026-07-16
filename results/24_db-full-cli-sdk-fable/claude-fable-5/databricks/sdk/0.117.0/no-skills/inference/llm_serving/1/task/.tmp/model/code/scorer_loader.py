"""Loader-module pyfunc wrapping the deterministic trigram scorer."""
import math

A = 2.449119
B = 1.093524
C = 1.192579
D = 1.949479


def _score(text):
    ll = 0.0
    for i in range(len(text) - 2):
        o0, o1, o2 = (ord(ch) for ch in text[i:i + 3])
        ll += math.sin(A * o0 + B * o1 + C * o2 + D)
    return round(ll, 6)


class TrigramScorer:
    def predict(self, model_input, params=None):
        try:
            texts = model_input["text"].tolist()
        except (TypeError, KeyError, IndexError, AttributeError):
            texts = [t for t in list(model_input)]
        return [_score(str(t)) for t in texts]


def _load_pyfunc(path):
    return TrigramScorer()
