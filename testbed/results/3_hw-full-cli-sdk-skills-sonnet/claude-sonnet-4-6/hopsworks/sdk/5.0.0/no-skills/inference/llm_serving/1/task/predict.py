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
    return round(ll, 6)


class Predict(object):
    def __init__(self):
        pass

    def predict(self, inputs):
        # inputs is the instances list: [[text], [text], ...] or [text, text, ...]
        # or the full request dict: {"instances": [...]}
        if isinstance(inputs, dict):
            instances = inputs.get("instances", [inputs])
        elif isinstance(inputs, list):
            instances = inputs
        else:
            instances = [inputs]

        predictions = []
        for instance in instances:
            if isinstance(instance, list) and len(instance) > 0:
                text = str(instance[0])
            elif isinstance(instance, str):
                text = instance
            else:
                text = str(instance)
            predictions.append(_score(text))

        return predictions
