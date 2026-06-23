import os
import sys
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


def _extract_texts(obj):
    """Pull the list of text strings out of whatever request shape arrives.
    Handles a bare string, a list of strings, a list wrapping a list, and
    dict bodies keyed by 'inputs' or 'instances'."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        for key in ("inputs", "instances", "data"):
            if key in obj:
                return _extract_texts(obj[key])
        return [str(obj)]
    if isinstance(obj, (list, tuple)):
        texts = []
        for item in obj:
            texts.extend(_extract_texts(item))
        return texts
    return [str(obj)]


class Predict:
    def __init__(self):
        path = load_model_file("scorer.py")
        spec = importlib.util.spec_from_file_location("scorer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._score = module.score

    def predict(self, inputs):
        texts = _extract_texts(inputs)
        scores = [self._score(t)["score"] for t in texts]
        return {
            "predictions": scores,
            "_dbg_type": str(type(inputs)),
            "_dbg_repr": str(inputs)[:300],
            "_dbg_texts": [str(t)[:80] for t in texts],
        }
