import os
import sys
import importlib.util


def _load_scorer():
    """Load the deterministic scorer from the downloaded model artifact.

    Hopsworks downloads the model files into the serving container and exposes
    the directory via the MODEL_FILES_PATH / ARTIFACT_FILES_PATH env vars.
    """
    candidates = []
    for var in ("MODEL_FILES_PATH", "ARTIFACT_FILES_PATH"):
        p = os.environ.get(var)
        if p:
            candidates.append(p)
    candidates.append(os.path.dirname(os.path.abspath(__file__)))

    for base in candidates:
        path = os.path.join(base, "scorer.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("scorer", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("scorer.py not found in model artifact")


class Predictor:
    def __init__(self):
        self._scorer = _load_scorer()

    def _score_one(self, text):
        return self._scorer.score(text)

    @staticmethod
    def _as_text(item):
        # An instance may arrive as a bare string, a single-element list
        # (list-of-lists request), or an object such as {"text": "..."}.
        if isinstance(item, str):
            return item
        if isinstance(item, (list, tuple)):
            return Predictor._as_text(item[0]) if item else ""
        if isinstance(item, dict):
            if "text" in item:
                return item["text"]
            for v in item.values():
                return Predictor._as_text(v)
            return ""
        return str(item)

    def predict(self, inputs):
        # KServe hands us the parsed request body. Accept a bare list, or
        # {"inputs": [...]} / {"instances": [...]}.
        data = inputs
        if isinstance(inputs, dict):
            if "inputs" in inputs:
                data = inputs["inputs"]
            elif "instances" in inputs:
                data = inputs["instances"]
        if isinstance(data, (str, dict)):
            data = [data]
        return [self._score_one(self._as_text(t)) for t in data]
