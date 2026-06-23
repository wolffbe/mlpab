import os
import glob
import importlib.util


def load_model_file(name):
    """Resolve a file saved alongside the model. Model files mount under
    MODEL_FILES_PATH at serving time; fall back to the standard mounts."""
    for root in (os.environ.get("MODEL_FILES_PATH"),
                 os.environ.get("ARTIFACT_FILES_PATH"),
                 "/mnt/models", "/mnt/artifacts"):
        if root:
            hits = glob.glob(f"{root}/**/{name}", recursive=True)
            if hits:
                return hits[0]
    raise FileNotFoundError(f"{name} not found under the model/artifact mounts")


class Predict:
    def __init__(self):
        path = load_model_file("scorer.py")
        spec = importlib.util.spec_from_file_location("scorer", path)
        self._scorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._scorer)

    def _to_text(self, item):
        # An instance may arrive as a bare string, a single-element list
        # (KServe "list of lists" shape), or an object with a text field.
        if isinstance(item, str):
            return item
        if isinstance(item, (list, tuple)):
            return self._to_text(item[0])
        if isinstance(item, dict):
            for key in ("text", "data", "payload", "input"):
                if key in item:
                    return self._to_text(item[key])
            return self._to_text(next(iter(item.values())))
        return str(item)

    def _score_one(self, item):
        return self._scorer.score(self._to_text(item))["score"]

    def predict(self, inputs):
        # `inputs` is the value under "instances" (or "inputs") in the request
        # body. Each instance carries one text payload.
        if isinstance(inputs, dict):
            inputs = inputs.get("instances", inputs.get("inputs", []))
        if isinstance(inputs, str):
            inputs = [inputs]
        predictions = [self._score_one(item) for item in inputs]
        return {"predictions": predictions}
