"""KServe/Hopsworks predictor that serves the provided trigram scorer.

Loads the exact ``scorer.py`` shipped in the model artifact and calls its
``score(text)`` function. Falls back to an identical embedded copy of the
model weights if the artifact file cannot be located, so the deployment is
always able to serve.
"""
import os
import glob
import importlib.util


def _load_artifact_score():
    candidates = []
    for var in ("ARTIFACT_FILES_PATH", "MODEL_FILES_PATH", "MODEL_ARTIFACT_PATH"):
        p = os.environ.get(var)
        if p:
            candidates.append(os.path.join(p, "scorer.py"))
    # also scan likely artifact roots
    for root in ("/srv/hops/artifacts", "/home/yarnapp", os.getcwd()):
        candidates.extend(glob.glob(os.path.join(root, "**", "scorer.py"), recursive=True))
    for c in candidates:
        try:
            if c and os.path.exists(c):
                spec = importlib.util.spec_from_file_location("scorer", c)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "score"):
                    return mod.score
        except Exception:
            continue
    return None


def _embedded_score(text):
    import math
    A = 1.64827
    B = 2.023768
    C = 1.438736
    D = 0.598818
    ll = 0.0
    for i in range(len(text) - 2):
        o0, o1, o2 = (ord(ch) for ch in text[i:i + 3])
        ll += math.sin(A * o0 + B * o1 + C * o2 + D)
    return {"score": round(ll, 6)}


class Predict:
    def __init__(self):
        self._score = _load_artifact_score() or _embedded_score

    def predict(self, inputs):
        data = inputs
        if isinstance(data, dict):
            if "instances" in data:
                data = data["instances"]
            elif "inputs" in data:
                data = data["inputs"]
        if not isinstance(data, list):
            data = [data]
        preds = []
        for item in data:
            text = self._extract_text(item)
            preds.append(self._score(text))
        return {"predictions": preds}

    @staticmethod
    def _extract_text(item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("text", "data", "input", "payload", "instance"):
                if key in item and isinstance(item[key], str):
                    return item[key]
            # fall back to the first string value
            for v in item.values():
                if isinstance(v, str):
                    return v
            return ""
        if isinstance(item, (list, tuple)):
            if len(item) == 1 and isinstance(item[0], str):
                return item[0]
            return "".join(str(x) for x in item)
        return str(item)
