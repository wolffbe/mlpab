import os
import importlib.util


def _load_scorer():
    """Locate scorer.py from the downloaded model artifact and import score()."""
    candidates = []
    for var in ("MODEL_FILES_PATH", "ARTIFACT_FILES_PATH", "MODEL_PATH"):
        p = os.environ.get(var)
        if p:
            candidates.append(p)
    candidates.append(os.getcwd())
    for base in candidates:
        for root, _dirs, files in os.walk(base):
            if "scorer.py" in files:
                path = os.path.join(root, "scorer.py")
                spec = importlib.util.spec_from_file_location("scorer", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.score
    raise RuntimeError("scorer.py not found in artifact; searched %s" % candidates)


class Predict:
    def __init__(self):
        self._score = _load_scorer()

    def _extract(self, inputs):
        # Unwrap common KServe envelopes down to a list of payload strings.
        if isinstance(inputs, dict):
            for k in ("instances", "inputs", "data"):
                if k in inputs:
                    return self._extract(inputs[k])
            return [inputs]
        if isinstance(inputs, list):
            return inputs
        return [inputs]

    def predict(self, inputs):
        texts = self._extract(inputs)
        results = []
        for t in texts:
            if isinstance(t, dict):
                # allow {"text": "..."} shaped instances
                t = t.get("text", next(iter(t.values())))
            results.append(self._score(str(t)))
        return {"predictions": results}
