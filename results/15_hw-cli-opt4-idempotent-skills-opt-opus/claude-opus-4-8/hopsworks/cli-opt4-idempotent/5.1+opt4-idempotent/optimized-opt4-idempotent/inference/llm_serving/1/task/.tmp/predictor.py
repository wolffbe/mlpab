"""KServe Python predictor that serves the provided deterministic scorer.

Loads `score(text)` from the registered model artifact (scorer.py). Falls back
to an identical embedded implementation if the artifact import path is not on
sys.path at runtime. Accepts KServe v1 bodies ({"instances": [...]}) as well as
a bare list / single string, and returns {"predictions": [...]}.
"""
import importlib.util
import math
import os
import sys


def _load_score():
    """Import score() from the model artifact's scorer.py if reachable."""
    candidates = []
    for var in ("ARTIFACT_FILES_PATH", "MODEL_FILES_PATH"):
        p = os.environ.get(var)
        if p:
            candidates.append(os.path.join(p, "scorer.py"))
    candidates.append(os.path.join(os.path.dirname(__file__), "scorer.py"))
    for path in candidates:
        if path and os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("scorer_artifact", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.score
    return None


# fixed model weights — identical to data/scorer.py (embedded fallback)
_A, _B, _C, _D = 0.83229, 0.59587, 1.017507, -1.194525


def _embedded_score(text):
    ll = 0.0
    for i in range(len(text) - 2):
        o0, o1, o2 = (ord(ch) for ch in text[i:i + 3])
        ll += math.sin(_A * o0 + _B * o1 + _C * o2 + _D)
    return {"score": round(ll, 6)}


class Predictor:
    def __init__(self):
        loaded = None
        try:
            loaded = _load_score()
        except Exception:
            loaded = None
        self._score = loaded if loaded is not None else _embedded_score

    @staticmethod
    def _to_text(item):
        # KServe requires list-of-lists or list-of-objects, so each instance
        # may be a ["text"] list, a {"text": ...}/{"data": ...} object, or a
        # bare string. Coerce any of these to the underlying text.
        if isinstance(item, str):
            return item
        if isinstance(item, (list, tuple)):
            return str(item[0]) if item else ""
        if isinstance(item, dict):
            for k in ("text", "data", "input", "payload", "value"):
                if k in item:
                    return str(item[k])
            vals = list(item.values())
            return str(vals[0]) if vals else ""
        return str(item)

    def predict(self, inputs):
        # Normalize KServe v1 / raw shapes into a list of instances.
        if isinstance(inputs, dict):
            data = inputs.get("instances", inputs.get("inputs", inputs))
        else:
            data = inputs
        if isinstance(data, (str, dict)):
            data = [data]
        elif not isinstance(data, (list, tuple)):
            data = [data]
        preds = [self._score(self._to_text(item)) for item in data]
        return {"predictions": preds}
