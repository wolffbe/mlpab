"""Custom predictor for the scorer model.

This module defines a predictor that calls the `score` function from scorer.py.
"""
from scorer import score


class Predictor:
    def predict(self, inputs):
        """Call the score function on the input text."""
        text = inputs['text']
        return score(text)