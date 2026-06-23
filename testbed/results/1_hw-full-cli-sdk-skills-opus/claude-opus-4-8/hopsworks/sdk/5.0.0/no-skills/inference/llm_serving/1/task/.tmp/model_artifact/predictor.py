import os
import sys

# The deployment artifact (including scorer.py) is downloaded to the path
# referenced by ARTIFACT_FILES_PATH / MODEL_FILES_PATH on the serving pod.
for _env in ("ARTIFACT_FILES_PATH", "MODEL_FILES_PATH"):
    _p = os.environ.get(_env)
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scorer


def _extract_text(inst):
    # KServe v1 wraps each instance; a single string payload arrives as ["text"].
    while isinstance(inst, (list, tuple)) and len(inst) == 1:
        inst = inst[0]
    if isinstance(inst, dict):
        # tolerate {"text": ...} / {"data": ...} shapes
        for k in ("text", "data", "input", "payload"):
            if k in inst:
                inst = inst[k]
                break
    if not isinstance(inst, str):
        inst = str(inst)
    return inst


class Predict:
    def __init__(self):
        self._score = scorer.score

    def predict(self, inputs):
        # `inputs` is the list of instances from the request body's "instances".
        if isinstance(inputs, dict):
            inputs = inputs.get("instances") or inputs.get("inputs") or []
        if isinstance(inputs, str):
            inputs = [inputs]
        results = []
        for inst in inputs:
            text = _extract_text(inst)
            results.append(self._score(text)["score"])
        return results
