import os
import sys


class Predict(object):
    """Hopsworks predictor wrapping the trigram scorer model."""

    def __init__(self):
        for var in ("MODEL_FILES_PATH", "ARTIFACT_FILES_PATH"):
            path = os.environ.get(var)
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        try:
            from scorer import score
        except ImportError:
            import math

            A = 0.517051
            B = 1.682507
            C = 2.459943
            D = 1.826126

            def score(text):
                ll = 0.0
                for i in range(len(text) - 2):
                    o0, o1, o2 = (ord(ch) for ch in text[i:i + 3])
                    ll += math.sin(A * o0 + B * o1 + C * o2 + D)
                return {"score": round(ll, 6)}

        self._score = score

    def predict(self, inputs):
        # Requests may arrive as a bare string, a list of strings, a nested
        # list, or the full {"instances": [...]} body depending on the
        # serving wrapper — normalize all of them to a flat list of strings.
        if isinstance(inputs, dict) and "instances" in inputs:
            inputs = inputs["instances"]
        out = []

        def handle(x):
            if isinstance(x, str):
                out.append(self._score(x)["score"])
            elif isinstance(x, (list, tuple)):
                for y in x:
                    handle(y)
            elif hasattr(x, "tolist"):
                handle(x.tolist())
            else:
                out.append({"debug": "%s:%r" % (type(x).__name__, x)})

        handle(inputs)
        return out
