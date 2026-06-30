"""A tiny deterministic pure-python "language model".

Character-trigram log-likelihood scorer: every trigram of the input text
contributes a weight derived from fixed constants; the score is the sum,
rounded to 6 decimal places. No dependencies beyond the standard library;
fully deterministic — the same text always yields the same score.

    >>> from scorer import score
    >>> score("hello world")
    {"score": ...}
"""
import json
import math
import sys

# model weights (fixed at training time — do not change)
A = 2.986326
B = 2.292444
C = 1.035017
D = -1.740277


def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)


def score(text):
    """Log-likelihood of `text` under the trigram model."""
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}


if __name__ == "__main__":
    payload = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    print(json.dumps(score(payload)))
