import os
import sys
import importlib.util


def _load_scorer():
    """Locate scorer.py inside the deployed model artifact and import it."""
    candidates = []
    for var in ("MODEL_FILES_PATH", "ARTIFACT_FILES_PATH"):
        p = os.environ.get(var)
        if p:
            candidates.append(p)
    candidates.append(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.getcwd())

    for base in candidates:
        for root, _dirs, files in os.walk(base):
            if "scorer.py" in files:
                spec = importlib.util.spec_from_file_location(
                    "scorer", os.path.join(root, "scorer.py")
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    raise RuntimeError("scorer.py not found in artifact paths: %r" % candidates)


class Predictor:
    def __init__(self):
        self._scorer = _load_scorer()

    def predict(self, inputs):
        # KServe passes the value of "instances" as `inputs` (a list).
        if isinstance(inputs, dict) and "instances" in inputs:
            inputs = inputs["instances"]
        if not isinstance(inputs, list):
            inputs = [inputs]
        results = []
        for item in inputs:
            if isinstance(item, str):
                text = item
            elif isinstance(item, list):
                text = item[0]
            elif isinstance(item, dict):
                text = item.get("text", next(iter(item.values())))
            else:
                text = str(item)
            results.append(self._scorer.score(text))
        return results
