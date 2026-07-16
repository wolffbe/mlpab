import math

A = 2.382892
B = 2.613343
C = 0.566573
D = -0.845423


def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)


def _score(text):
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}


def _extract_text(item):
    """Unwrap text from various container formats."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text", str(item))
    if isinstance(item, (list, tuple)):
        return _extract_text(item[0]) if item else ""
    return str(item)


class Predict(object):
    def __init__(self):
        pass

    def predict(self, inputs):
        if isinstance(inputs, dict):
            instances = inputs.get("instances", [])
            results = [_score(_extract_text(item)) for item in instances]
            return {"predictions": results}
        else:
            # single instance delivered directly
            return _score(_extract_text(inputs))
