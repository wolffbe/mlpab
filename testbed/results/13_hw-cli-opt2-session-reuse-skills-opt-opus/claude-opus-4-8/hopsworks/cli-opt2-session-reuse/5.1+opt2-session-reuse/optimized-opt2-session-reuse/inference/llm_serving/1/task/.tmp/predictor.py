"""KServe Python predictor that serves the provided deterministic scorer.

It loads `scorer.py` from the downloaded model artifact directory and calls
its `score(text)` function. A verbatim copy of the scorer is embedded as a
fallback so the endpoint serves correct, identical results even if the
artifact path differs across runtime versions (the function is deterministic
and stdlib-only, so the embedded copy yields the same scores).
"""
import importlib.util
import math
import os


# --- verbatim fallback copy of the provided scorer (stdlib only) ---
_A = 1.674569
_B = 0.682464
_C = 1.602544
_D = 1.292792


def _fallback_score(text):
    ll = 0.0
    for i in range(len(text) - 2):
        tri = text[i:i + 3]
        o0, o1, o2 = (ord(ch) for ch in tri)
        ll += math.sin(_A * o0 + _B * o1 + _C * o2 + _D)
    return {"score": round(ll, 6)}


def _find_scorer():
    """Locate scorer.py inside the downloaded model artifact, if present."""
    candidates = []
    for var in ("MODEL_FILES_PATH", "ARTIFACT_FILES_PATH", "MODEL_PATH"):
        p = os.environ.get(var)
        if p:
            candidates.append(p)
    candidates.append(os.getcwd())
    for base in candidates:
        for root, _dirs, files in os.walk(base):
            if "scorer.py" in files:
                return os.path.join(root, "scorer.py")
    return None


class Predict:
    def __init__(self):
        self._score = _fallback_score
        try:
            path = _find_scorer()
            if path:
                spec = importlib.util.spec_from_file_location("scorer", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "score"):
                    self._score = mod.score
        except Exception:
            # fall back to the embedded copy on any load error
            self._score = _fallback_score

    @staticmethod
    def _to_text(item):
        # An instance may arrive as a bare string, a single-element list
        # (KServe "list of lists"), or an object (KServe "list of objects").
        if isinstance(item, str):
            return item
        if isinstance(item, (list, tuple)):
            return Predict._to_text(item[0]) if item else ""
        if isinstance(item, dict):
            for key in ("text", "input", "data", "payload", "instance"):
                if key in item and isinstance(item[key], str):
                    return item[key]
            for val in item.values():
                if isinstance(val, str):
                    return val
            return ""
        return str(item)

    def predict(self, inputs):
        # KServe passes the request body; accept dict-with-instances/inputs,
        # a bare list, or a single string.
        if isinstance(inputs, dict):
            data = inputs.get("instances", inputs.get("inputs", inputs))
        else:
            data = inputs
        if isinstance(data, str):
            data = [data]
        results = [self._score(self._to_text(item)) for item in data]
        return {"predictions": results}
