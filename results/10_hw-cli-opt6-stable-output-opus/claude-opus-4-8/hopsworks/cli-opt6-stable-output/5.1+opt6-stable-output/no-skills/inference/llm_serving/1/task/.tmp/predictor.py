import os
import sys
import math


# model weights (fixed at training time — do not change), mirrored from scorer.py
A = 2.467614
B = 0.585566
C = 0.845413
D = -1.682077


def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)


def _score(text):
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}


class Predict(object):
    def __init__(self):
        # Prefer the registered scorer artifact if importable; otherwise use the
        # mirrored implementation above (identical deterministic logic).
        self._score_fn = _score
        artifact_path = os.environ.get("ARTIFACT_FILES_PATH") or os.environ.get(
            "MODEL_FILES_PATH"
        )
        if artifact_path and os.path.isdir(artifact_path):
            try:
                if artifact_path not in sys.path:
                    sys.path.insert(0, artifact_path)
                import scorer as _scorer  # noqa

                self._score_fn = _scorer.score
            except Exception:
                self._score_fn = _score

    @staticmethod
    def _extract_text(item):
        # Accept several KServe instance shapes:
        #   "text"                      -> bare string
        #   ["text"]                    -> list of lists
        #   {"text": "..."} / {"data": "..."} / single-value object
        if isinstance(item, str):
            return item
        if isinstance(item, list):
            return Predict._extract_text(item[0]) if item else ""
        if isinstance(item, dict):
            for key in ("text", "data", "input", "payload"):
                if key in item:
                    return Predict._extract_text(item[key])
            values = list(item.values())
            return Predict._extract_text(values[0]) if values else ""
        return str(item)

    def predict(self, inputs):
        # KServe payload: {"instances": [...]} -> inputs is the list of instances.
        if isinstance(inputs, dict) and "instances" in inputs:
            inputs = inputs["instances"]
        if not isinstance(inputs, list):
            inputs = [inputs]
        predictions = []
        for item in inputs:
            predictions.append(self._score_fn(self._extract_text(item)))
        return {"predictions": predictions}
