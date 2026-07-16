import os
import glob
import importlib.util


def load_model_file(name):
    """Resolve a file saved alongside the model. Model files mount under
    MODEL_FILES_PATH at serving time (ARTIFACT_FILES_PATH holds only this
    script); fall back to the standard mount roots."""
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
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._score = module.score

    def predict(self, inputs):
        # `inputs` is the value of the "inputs" key from the request body.
        # Accept a single string or a list of strings.
        if isinstance(inputs, str):
            texts = [inputs]
        else:
            texts = list(inputs)
        scores = [self._score(t)["score"] for t in texts]
        return {"predictions": scores}
