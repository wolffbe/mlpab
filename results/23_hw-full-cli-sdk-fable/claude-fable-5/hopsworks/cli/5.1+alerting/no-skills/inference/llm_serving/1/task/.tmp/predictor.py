import os
import sys


class Predict(object):
    """Serves the pure-python trigram scorer from the model artifact."""

    def __init__(self):
        candidates = [
            os.environ.get("MODEL_FILES_PATH"),
            os.environ.get("ARTIFACT_FILES_PATH"),
            os.getcwd(),
        ]
        scorer_dir = None
        for base in candidates:
            if not base or not os.path.isdir(base):
                continue
            for root, _dirs, files in os.walk(base):
                if "scorer.py" in files:
                    scorer_dir = root
                    break
            if scorer_dir:
                break
        if scorer_dir is None:
            raise RuntimeError("scorer.py not found in model files")
        sys.path.insert(0, scorer_dir)
        import scorer

        self._score = scorer.score

    def predict(self, inputs):
        results = []
        for item in inputs:
            if isinstance(item, dict):
                text = item.get("text", "")
            elif isinstance(item, (list, tuple)):
                text = item[0]
            else:
                text = item
            results.append(self._score(text))
        return results
