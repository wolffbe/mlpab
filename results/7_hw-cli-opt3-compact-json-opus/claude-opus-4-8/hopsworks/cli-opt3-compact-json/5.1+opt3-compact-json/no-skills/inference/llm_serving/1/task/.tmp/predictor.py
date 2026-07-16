import os
import math
import importlib.util

# Model weights — identical to the provided data/scorer.py (fixed at training time).
A = 1.626184
B = 0.684995
C = 1.964646
D = -0.634848


def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)


def _inline_score(text):
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}


def _load_provided_score():
    """Prefer the provided scorer.py shipped inside the model artifact."""
    candidates = [
        os.environ.get("ARTIFACT_FILES_PATH"),
        os.environ.get("MODEL_FILES_PATH"),
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for base in candidates:
        if not base:
            continue
        path = os.path.join(base, "scorer.py")
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location("provided_scorer", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "score"):
                    return module.score
            except Exception:
                pass
    return None


class Predictor:
    def __init__(self):
        self._score = _load_provided_score() or _inline_score

    @staticmethod
    def _as_text(instance):
        # Normalize one instance to the text string to score.
        if isinstance(instance, str):
            return instance
        if isinstance(instance, list):
            # list of one (or more) — score the first element as text.
            return instance[0] if instance else ""
        if isinstance(instance, dict):
            for key in ("text", "input", "payload", "data"):
                if key in instance:
                    return instance[key]
            # fall back to the first value
            vals = list(instance.values())
            return vals[0] if vals else ""
        return str(instance)

    def predict(self, inputs):
        # inputs is the parsed value of the request body's "instances" key.
        if isinstance(inputs, dict):
            inputs = inputs.get("instances", inputs)
        if isinstance(inputs, str):
            inputs = [inputs]
        results = []
        for instance in inputs:
            results.append(self._score(self._as_text(instance)))
        return results
