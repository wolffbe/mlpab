import os
import sys


class Predict:
    """KServe Python predictor that serves the deterministic trigram scorer."""

    def __init__(self):
        # The registered model's files are downloaded to this path on the
        # serving pod; add it to sys.path so we can import scorer.py.
        model_path = (
            os.environ.get("MODEL_FILES_PATH")
            or os.environ.get("ARTIFACT_FILES_PATH")
            or os.environ.get("MODEL_PATH")
            or "."
        )
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
        import scorer  # noqa: E402

        self._score = scorer.score

    @staticmethod
    def _as_text(instance):
        # KServe accepts instances as a list of lists or a list of objects.
        # Normalize each instance back to the raw text the scorer expects.
        if isinstance(instance, str):
            return instance
        if isinstance(instance, list):
            return instance[0]
        if isinstance(instance, dict):
            for key in ("text", "input", "payload", "data"):
                if key in instance:
                    return instance[key]
            return next(iter(instance.values()))
        return str(instance)

    def predict(self, inputs):
        # `inputs` is the value of the "instances" key in the request body.
        # Return one score dict per payload, in order.
        if isinstance(inputs, dict):
            inputs = inputs.get("instances", inputs)
        return [self._score(self._as_text(item)) for item in inputs]
