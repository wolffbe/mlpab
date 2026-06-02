"""mlkit CLI: a fake AutoML CLI over a local platform.

    mlkit login
    mlkit fit --data data                       # -> model_id=<id>
    mlkit predict --model <id> --data data --out submission/submission.csv
"""
from __future__ import annotations
import argparse

from mlkit_cli import _client


def main(argv=None):
    p = argparse.ArgumentParser(prog="mlkit", description="mlkit fake AutoML CLI")
    p.add_argument("--version", action="version", version="mlkit 0.1.0")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("login", help="authenticate with MLKIT_API_KEY")
    f = sub.add_parser("fit", help="train a model on <data>/train.csv")
    f.add_argument("--data", default="data")
    pr = sub.add_parser("predict", help="write a submission from a fitted model")
    pr.add_argument("--model", default=None, help="model_id from `mlkit fit`")
    pr.add_argument("--data", default="data")
    pr.add_argument("--out", default="submission/submission.csv")
    a = p.parse_args(argv)
    if a.cmd == "login":
        _client.login()
        print("[mlkit] logged in")
    elif a.cmd == "fit":
        print(_client.fit(a.data))
    elif a.cmd == "predict":
        print(_client.predict(a.model, a.data, a.out))
    else:
        p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
