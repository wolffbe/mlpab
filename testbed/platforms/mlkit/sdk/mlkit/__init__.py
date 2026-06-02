"""mlkit SDK: a fake AutoML library over a local platform.

    import mlkit
    mlkit.login()
    info = mlkit.fit("data")                                   # info["model_id"]
    mlkit.predict(info["model_id"], "data", "submission/submission.csv")
"""
from mlkit._client import login, fit, predict

__version__ = "0.1.0"
__all__ = ["login", "fit", "predict", "__version__"]
