import os
import sys
import glob
import importlib.util


def _load_scorer():
    for root in (os.environ.get("MODEL_FILES_PATH"),
                 os.environ.get("ARTIFACT_FILES_PATH"),
                 "/mnt/models", "/mnt/artifacts"):
        if root:
            hits = glob.glob(f"{root}/**/scorer.py", recursive=True)
            if hits:
                spec = importlib.util.spec_from_file_location("scorer", hits[0])
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    raise FileNotFoundError("scorer.py not found under model/artifact mounts")


class Predict:
    def __init__(self):
        self.scorer = _load_scorer()

    def predict(self, inputs, feature_vectors=None):
        results = []
        if isinstance(inputs, list):
            for item in inputs:
                if isinstance(item, dict):
                    text = item.get("text", item.get("instances", ""))
                    if isinstance(text, list):
                        text = text[0] if text else ""
                else:
                    text = str(item)
                results.append(self.scorer.score(text))
        elif isinstance(inputs, dict):
            text = inputs.get("text", "")
            results.append(self.scorer.score(text))
        else:
            results.append(self.scorer.score(str(inputs)))
        return {"predictions": results}
