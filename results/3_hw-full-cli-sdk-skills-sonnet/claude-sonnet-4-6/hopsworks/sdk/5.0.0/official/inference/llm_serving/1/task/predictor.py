import os
import glob
import importlib.util


def _load_model_file(name):
    for root in (os.environ.get("MODEL_FILES_PATH"),
                 os.environ.get("ARTIFACT_FILES_PATH"),
                 "/mnt/models", "/mnt/artifacts"):
        if root:
            hits = glob.glob(f"{root}/**/{name}", recursive=True)
            if hits:
                return hits[0]
    raise FileNotFoundError(f"{name} not found under model/artifact mounts")


class Predict:
    def __init__(self):
        scorer_path = _load_model_file("scorer.py")
        spec = importlib.util.spec_from_file_location("scorer", scorer_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._score = mod.score

    def predict(self, inputs):
        results = []
        for item in inputs:
            if isinstance(item, dict):
                text = item.get("text", item.get("instances", ""))
            else:
                text = str(item)
            results.append(self._score(text)["score"])
        return {"predictions": results}
