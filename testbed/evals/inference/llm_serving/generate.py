"""LLM-serving task (FTI sub-category: inference/llm_serving) — generator.

Usage:
    python -m evals.inference.llm_serving.generate --seed 7 --out /tmp/llm-7
    python -m evals.inference.llm_serving.generate --selftest

World: a PROVIDED deterministic pure-python "language model"
(`data/scorer.py` — a character-trigram log-likelihood scorer whose weights
are derived from the seed; `score(text)` returns {"score": round(ll, 6)}),
plus 5 seeded text payloads (`data/payloads.json`). The agent must deploy the
scorer as a REAL-TIME ENDPOINT named `scorer<sfx>` (per-instance suffix) on
the platform, invoke it on each payload, and write `submission/answers.json`:

    {"endpoint_name": "scorer<sfx>", "responses": [<score per payload, in order>]}

Ground truth by construction: the generator imports the very scorer module it
wrote and computes the 5 scores. Generation-time gates run the grade function
(adapter `none`): the reference answers pass; a perturbed response and a
wrong endpoint name both fail.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from evals.common import instance_suffix

KIND = "platform"  # deliverable kind: table | dataset | answers | platform
SUMMARY = (
    "Measures whether an agent can deploy a provided model as a real-time endpoint on the "
    "platform and invoke it correctly over the platform's inference path."
)

ENDPOINT_BASE = "scorer"  # per-instance: f"{ENDPOINT_BASE}{instance_suffix(seed)}"
N_PAYLOADS = 5
VOCAB = [
    "feature",
    "store",
    "online",
    "vector",
    "pipeline",
    "model",
    "drift",
    "lookup",
    "serving",
    "batch",
    "training",
    "embedding",
    "latency",
    "monitor",
    "schedule",
    "registry",
    "inference",
    "stream",
]

SCORER_TEMPLATE = '''"""A tiny deterministic pure-python "language model".

Character-trigram log-likelihood scorer: every trigram of the input text
contributes a weight derived from fixed constants; the score is the sum,
rounded to 6 decimal places. No dependencies beyond the standard library;
fully deterministic — the same text always yields the same score.

    >>> from scorer import score
    >>> score("hello world")
    {{"score": ...}}
"""
import json
import math
import sys

# model weights (fixed at training time — do not change)
A = {a!r}
B = {b!r}
C = {c!r}
D = {d!r}


def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)


def score(text):
    """Log-likelihood of `text` under the trigram model."""
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {{"score": round(ll, 6)}}


if __name__ == "__main__":
    payload = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    print(json.dumps(score(payload)))
'''


class GateError(RuntimeError):
    pass


def _load_scorer(path: Path):
    spec = importlib.util.spec_from_file_location(f"mlpab_scorer_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    endpoint = ENDPOINT_BASE + instance_suffix(seed)
    scorer_src = SCORER_TEMPLATE.format(
        a=round(float(rng.uniform(0.5, 3.0)), 6),
        b=round(float(rng.uniform(0.5, 3.0)), 6),
        c=round(float(rng.uniform(0.5, 3.0)), 6),
        d=round(float(rng.uniform(-2.0, 2.0)), 6),
    )
    payloads = [" ".join(rng.choice(VOCAB, int(rng.integers(6, 13)))) for _ in range(N_PAYLOADS)]

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    (out / "data" / "scorer.py").write_text(scorer_src)
    (out / "data" / "payloads.json").write_text(json.dumps(payloads, indent=2))

    # truth: import the very module we wrote and score the payloads with it
    scorer = _load_scorer(out / "data" / "scorer.py")
    responses = [scorer.score(p)["score"] for p in payloads]

    (out / "prompt.txt").write_text(
        "The directory data/ contains a small deterministic pure-python "
        "language-model scorer (data/scorer.py — `score(text)` returns "
        '{"score": <float>}; standard library only) and 5 text payloads '
        "(data/payloads.json, a JSON list of strings).\n"
        f"Deploy the provided scorer as a REAL-TIME ENDPOINT named `{endpoint}` "
        "on the platform (a live endpoint/deployment that serves the scorer "
        "over the platform's inference path).\n"
        "Invoke the endpoint on each payload, in the order given, and write "
        "submission/answers.json as:\n"
        f'    {{"endpoint_name": "{endpoint}", "responses": [<the 5 scores, '
        "in payload order>]}\n"
    )
    truth = {
        "family": "llm_serving",
        "seed": seed,
        "endpoint_name": endpoint,
        "payloads": payloads,
        "responses": responses,
    }
    (out / "solution" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "instance.json").write_text(
        json.dumps({"family": "llm_serving", "seed": seed}, indent=2)
    )

    # --- gates: the grade function accepts truth, rejects corruptions ----------
    from evals.inference.llm_serving.grade import grade

    def run(answers: dict) -> bool:
        with tempfile.TemporaryDirectory(prefix="mlpab-llm-gate-") as td:
            run_dir = Path(td)
            (run_dir / "submission").mkdir()
            (run_dir / "submission" / "answers.json").write_text(json.dumps(answers))
            return grade(out, "none", run_dir)["success"]

    if not run({"endpoint_name": endpoint, "responses": responses}):
        raise GateError(f"reference answers fail the grade function (seed={seed})")
    perturbed = list(responses)
    perturbed[0] = round(perturbed[0] + 0.001, 6)
    if run({"endpoint_name": endpoint, "responses": perturbed}):
        raise GateError(f"perturbed response passes the grade function (seed={seed})")
    if run({"endpoint_name": "my_endpoint", "responses": responses}):
        raise GateError(f"wrong endpoint name passes the grade function (seed={seed})")
    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-llm-selftest/{seed}"))
            print(f"[llm_serving] seed={seed} payloads={len(meta['payloads'])} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
