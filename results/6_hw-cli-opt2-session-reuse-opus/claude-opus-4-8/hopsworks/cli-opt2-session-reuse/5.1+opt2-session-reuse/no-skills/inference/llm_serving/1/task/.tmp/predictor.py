import os
import sys
import math
import importlib.util


def _load_score():
    """Load score() from the registered scorer.py artifact if available,
    otherwise fall back to an embedded identical copy."""
    candidates = []
    for var in ("ARTIFACT_FILES_PATH", "MODEL_FILES_PATH"):
        p = os.environ.get(var)
        if p:
            candidates.append(p)
    candidates.append(os.path.dirname(os.path.abspath(__file__)))
    for base in candidates:
        cand = os.path.join(base, "scorer.py")
        if os.path.exists(cand):
            try:
                spec = importlib.util.spec_from_file_location("scorer", cand)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "score"):
                    return mod.score
            except Exception:
                pass
    # Embedded fallback — byte-for-byte identical to data/scorer.py logic.
    A = 2.313042
    B = 2.631279
    C = 1.640452
    D = -0.995959

    def score(text):
        ll = 0.0
        for i in range(len(text) - 2):
            o0, o1, o2 = (ord(ch) for ch in text[i:i + 3])
            ll += math.sin(A * o0 + B * o1 + C * o2 + D)
        return {"score": round(ll, 6)}

    return score


class Predict(object):
    def __init__(self):
        self._score = _load_score()

    def _score_one(self, text):
        return self._score(text)["score"]

    def predict(self, inputs):
        # KServe passes the request body; Hopsworks commonly wraps the user
        # payload under an "instances" key. Accept several shapes and always
        # return a list of scores under "predictions".
        data = inputs
        if isinstance(data, dict):
            if "instances" in data:
                data = data["instances"]
            elif "inputs" in data:
                data = data["inputs"]

        if isinstance(data, str):
            items = [data]
        elif isinstance(data, list):
            items = data
        else:
            items = [data]

        preds = []
        for it in items:
            if isinstance(it, dict):
                # try common text keys
                text = it.get("text", it.get("input", next(iter(it.values()))))
            else:
                text = it
            preds.append(self._score_one(str(text)))
        return {"predictions": preds}
