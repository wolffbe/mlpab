import os
import sys
import math

# Embedded copy of the provided scorer (deterministic, stdlib-only).
# The artifact also ships scorer.py; we prefer importing it, with this as fallback.
_A = 1.56271
_B = 2.959377
_C = 2.770975
_D = 0.367676


def _embedded_score(text):
    ll = 0.0
    for i in range(len(text) - 2):
        o0, o1, o2 = (ord(ch) for ch in text[i:i + 3])
        ll += math.sin(_A * o0 + _B * o1 + _C * o2 + _D)
    return {"score": round(ll, 6)}


class Predict:
    def __init__(self):
        self._score_fn = _embedded_score
        # Try to load the provided scorer.py from the downloaded model artifact.
        candidates = []
        for var in ("MODEL_FILES_PATH", "ARTIFACT_FILES_PATH", "MODEL_PATH"):
            p = os.environ.get(var)
            if p:
                candidates.append(p)
        for base in candidates:
            try:
                if base not in sys.path:
                    sys.path.insert(0, base)
                import scorer as _scorer  # noqa
                if hasattr(_scorer, "score"):
                    self._score_fn = _scorer.score
                    break
            except Exception:
                continue

    def predict(self, inputs):
        # inputs is the "instances" list from the request body.
        # Each instance is a text string; return a score dict per instance.
        if isinstance(inputs, dict):
            inputs = inputs.get("instances", inputs)
        if isinstance(inputs, str):
            inputs = [inputs]
        results = []
        for item in inputs:
            text = self._extract_text(item)
            results.append(self._score_fn(text))
        return results

    @staticmethod
    def _extract_text(item):
        # Instances may arrive as a bare string, a one-element list (["text"]),
        # or an object ({"text": "..."} / {"instances": "..."}).
        if isinstance(item, str):
            return item
        if isinstance(item, (list, tuple)):
            return Predict._extract_text(item[0]) if item else ""
        if isinstance(item, dict):
            for key in ("text", "data", "input", "payload", "instances"):
                if key in item:
                    return Predict._extract_text(item[key])
            vals = list(item.values())
            return Predict._extract_text(vals[0]) if vals else ""
        return str(item)
