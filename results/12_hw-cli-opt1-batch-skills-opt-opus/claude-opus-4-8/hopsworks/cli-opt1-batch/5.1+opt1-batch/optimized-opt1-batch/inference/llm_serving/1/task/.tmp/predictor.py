import os
import sys
import glob


def _find_scorer_dir():
    """Locate the directory containing the registered scorer.py on the
    serving pod. Hopsworks downloads the model artifact to MODEL_FILES_PATH
    (and exposes ARTIFACT_FILES_PATH); search both, then fall back to cwd."""
    bases = [
        os.environ.get("MODEL_FILES_PATH"),
        os.environ.get("ARTIFACT_FILES_PATH"),
        os.getcwd(),
    ]
    for base in bases:
        if not base:
            continue
        if os.path.exists(os.path.join(base, "scorer.py")):
            return base
        hits = glob.glob(os.path.join(base, "**", "scorer.py"), recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None


class Predict(object):
    def __init__(self):
        d = _find_scorer_dir()
        if d and d not in sys.path:
            sys.path.insert(0, d)
        import scorer
        self._score = scorer.score

    def predict(self, inputs):
        """KServe v1 protocol: `inputs` is the list under the request's
        "instances" key. Each instance is a text payload to score. Returns
        the list of float scores (one per instance), in order."""
        instances = inputs
        if isinstance(inputs, dict) and "instances" in inputs:
            instances = inputs["instances"]
        if not isinstance(instances, list):
            instances = [instances]
        results = []
        for item in instances:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text", item.get("payload", ""))
            else:
                text = str(item)
            results.append(self._score(text)["score"])
        return results
