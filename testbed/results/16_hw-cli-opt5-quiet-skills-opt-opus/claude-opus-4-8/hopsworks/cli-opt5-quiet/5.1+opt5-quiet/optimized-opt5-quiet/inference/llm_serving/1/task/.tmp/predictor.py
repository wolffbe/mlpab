import os
import sys


class Predictor:
    """KServe python predictor that serves the provided trigram scorer."""

    def __init__(self):
        # scorer.py ships as a model-registry artifact; on the serving pod it
        # may land in any of several mount points. Locate it robustly.
        candidates = []
        for var in ("ARTIFACT_FILES_PATH", "MODEL_FILES_PATH"):
            p = os.environ.get(var)
            if p:
                candidates.append(p)
        candidates.append(os.path.dirname(os.path.abspath(__file__)))

        scorer_dir = None
        for root in candidates:
            if root and os.path.isfile(os.path.join(root, "scorer.py")):
                scorer_dir = root
                break
        if scorer_dir is None:
            # Walk common mount roots to find scorer.py.
            search_roots = candidates + ["/mnt", "/srv"]
            for root in search_roots:
                if not root or not os.path.isdir(root):
                    continue
                for dirpath, _dirs, files in os.walk(root):
                    if "scorer.py" in files:
                        scorer_dir = dirpath
                        break
                if scorer_dir:
                    break
        if scorer_dir and scorer_dir not in sys.path:
            sys.path.insert(0, scorer_dir)
        import scorer

        self._score = scorer.score

    def predict(self, inputs):
        # Accept either the raw list of texts or a KServe-style request body
        # {"instances": [...]}. Return one score float per input, in order.
        if isinstance(inputs, dict) and "instances" in inputs:
            items = inputs["instances"]
        else:
            items = inputs
        if not isinstance(items, list):
            items = [items]

        def _as_text(item):
            # KServe requires instances be lists/objects, so a text payload
            # arrives wrapped as ["text"] or {"text": "..."}; unwrap it.
            if isinstance(item, list):
                return item[0] if item else ""
            if isinstance(item, dict):
                if "text" in item:
                    return item["text"]
                vals = list(item.values())
                return vals[0] if vals else ""
            return item

        return [self._score(_as_text(it))["score"] for it in items]
