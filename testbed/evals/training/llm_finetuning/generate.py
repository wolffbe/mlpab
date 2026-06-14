"""LLM fine-tuning task (FTI sub-category: training/finetune) — generator.

Usage:
    python -m evals.training.llm_finetuning.generate --seed 7 --out /tmp/llm_finetuning-7
    python -m evals.training.llm_finetuning.generate --selftest

The world provides a base character-level language model checkpoint
(data/base_model.npz — a bigram logit matrix over a 28-char vocab, produced
by deterministically "pre-training" on a seeded base corpus inside this
generator), a fine-tuning corpus (data/finetune.txt) sampled from a seeded
Markov chain whose distribution is SHIFTED vs the base corpus, a held-out
eval set from the same shifted distribution (data/eval.txt), and
data/finetune_model.py — a fully deterministic pure numpy fine-tuning script
(LoRA-style parameter-efficient update: base logits frozen, only a rank-4
additive adapter delta = A @ B is trained by plain full-batch gradient
descent, learning rate 2.0, exactly 400 iterations; B starts at zero so the
adapter delta starts at zero, A is a fixed deterministic tiled identity —
both factors at zero would be a dead saddle with exactly-zero gradients)
that writes finetuned_model.npz and metrics.json
{"eval_loss": …, "base_eval_loss": …} (cross-entropy on eval.txt, 4 dp).
The agent must run it AS A JOB ON THE PLATFORM (named `ftjob<sfx>`), register
the fine-tuned model as `ftmodel<sfx>` v1 with the metrics attached, and
report names + metrics in answers.json.

Ground truth by construction: the generator reimplements the identical
fine-tuning math inline; a gate additionally runs the emitted
finetune_model.py via subprocess in a temp dir and requires it to reproduce
the truth metrics EXACTLY. Fine-tuning must actually help (eval_loss <
base_eval_loss by a clear margin) and — same artifact class as odt's
rounding-tie fix — every pre-rounding loss must sit >= 1e-6 away from a 4-dp
rounding boundary; worlds violating either are deterministically resampled
from seed-derived sub-seeds. Naive variants (gates assert they differ from
truth): reporting base_eval_loss as eval_loss (no_finetune — the agent
skipped the fine-tune) and training for only 40 iterations (undertrained).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from evals.common import instance_suffix
from evals.training.llm_finetuning.grade import grade_answers

VOCAB = list("abcdefghijklmnopqrstuvwxyz .")
RANK = 4
LEARNING_RATE = 2.0
ITERATIONS = 400
ADAPTER_SCALE = 0.5
N_BASE = 60_000  # base ("pre-training") corpus — consumed in-generator only
N_FINETUNE = 30_000  # data/finetune.txt
N_EVAL = 8_000  # data/eval.txt
MIN_IMPROVEMENT = 0.05  # pre-rounding: base_eval_loss - eval_loss must exceed this
TIE_MARGIN = 1e-6  # min distance of a pre-rounding loss from a 4-dp boundary
MAX_RESAMPLES = 64
JOB_BASE = "ftjob"  # per-instance: f"{JOB_BASE}{instance_suffix(seed)}"
MODEL_BASE = "ftmodel"  # per-instance: f"{MODEL_BASE}{instance_suffix(seed)}"
VERSION = 1

VARIANT_DIAGNOSIS = {
    "no_finetune": "eval_loss is the UNADAPTED base model's loss — the "
    "fine-tune was skipped (or its output ignored)",
    "undertrained": "eval_loss comes from an undertrained adapter "
    "(40 iterations instead of the script's 400)",
}

FINETUNE_SCRIPT = '''"""Provided fine-tuning script — deterministic LoRA-style adapter update.

Reads base_model.npz (bigram logits + vocab), finetune.txt and eval.txt from
its working directory. The base logits stay FROZEN; only a rank-4 additive
adapter delta = A @ B is trained by plain full-batch gradient descent
(learning rate 2.0, exactly 400 iterations; B initialized to zero so the
adapter delta starts at zero, A to a fixed 0.5-scaled tiled identity).
Writes finetuned_model.npz and metrics.json with the fine-tuned and unadapted
cross-entropy on eval.txt (both rounded to 4 decimals). Fully deterministic —
do NOT modify it; run it as-is.
"""
import json

import numpy as np

RANK = 4
LEARNING_RATE = 2.0
ITERATIONS = 400
ADAPTER_SCALE = 0.5


def bigram_counts(text, vocab):
    index = {ch: i for i, ch in enumerate(vocab)}
    idx = np.array([index[ch] for ch in text], dtype=int)
    counts = np.zeros((len(vocab), len(vocab)), dtype=float)
    np.add.at(counts, (idx[:-1], idx[1:]), 1.0)
    return counts


def cross_entropy(logits, counts):
    logp = logits - logits.max(axis=1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))
    return -(counts * logp).sum() / counts.sum()


def main():
    ckpt = np.load("base_model.npz")
    base_logits = ckpt["logits"].astype(float)
    vocab = [str(ch) for ch in ckpt["vocab"]]
    ft_counts = bigram_counts(open("finetune.txt").read(), vocab)
    eval_counts = bigram_counts(open("eval.txt").read(), vocab)

    n_vocab = base_logits.shape[0]
    A = np.zeros((n_vocab, RANK), dtype=float)
    for i in range(n_vocab):
        A[i, i % RANK] = ADAPTER_SCALE
    B = np.zeros((RANK, n_vocab), dtype=float)
    n = ft_counts.sum()
    row_totals = ft_counts.sum(axis=1, keepdims=True)
    for _ in range(ITERATIONS):
        logits = base_logits + A @ B
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        g = (row_totals * p - ft_counts) / n
        A, B = A - LEARNING_RATE * (g @ B.T), B - LEARNING_RATE * (A.T @ g)

    np.savez("finetuned_model.npz", logits=base_logits + A @ B,
             adapter_a=A, adapter_b=B)
    metrics = {
        "eval_loss": round(float(cross_entropy(base_logits + A @ B, eval_counts)), 4),
        "base_eval_loss": round(float(cross_entropy(base_logits, eval_counts)), 4),
    }
    json.dump(metrics, open("metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
'''


class GateError(RuntimeError):
    pass


# --- the identical fine-tuning math, reimplemented inline (see FINETUNE_SCRIPT) ---


def _bigram_counts(idx: np.ndarray) -> np.ndarray:
    counts = np.zeros((len(VOCAB), len(VOCAB)), dtype=float)
    np.add.at(counts, (idx[:-1], idx[1:]), 1.0)
    return counts


def _cross_entropy(logits: np.ndarray, counts: np.ndarray) -> float:
    logp = logits - logits.max(axis=1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))
    return float(-(counts * logp).sum() / counts.sum())


def _finetune(base_logits: np.ndarray, ft_counts: np.ndarray, iterations: int) -> np.ndarray:
    """Adapter training — expression-for-expression the script's loop, so the
    inline truth and the emitted script are bit-identical."""
    n_vocab = base_logits.shape[0]
    A = np.zeros((n_vocab, RANK), dtype=float)
    for i in range(n_vocab):
        A[i, i % RANK] = ADAPTER_SCALE
    B = np.zeros((RANK, n_vocab), dtype=float)
    n = ft_counts.sum()
    row_totals = ft_counts.sum(axis=1, keepdims=True)
    for _ in range(iterations):
        logits = base_logits + A @ B
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        g = (row_totals * p - ft_counts) / n
        A, B = A - LEARNING_RATE * (g @ B.T), B - LEARNING_RATE * (A.T @ g)
    return base_logits + A @ B


def _sample_chain(rng: np.random.Generator, trans: np.ndarray, n: int) -> np.ndarray:
    """n-step Markov walk over vocab indices (row-stochastic `trans`)."""
    cum = np.cumsum(trans, axis=1)
    u = rng.random(n)
    out = np.empty(n, dtype=int)
    cur = 0
    for i in range(n):
        cur = int(np.searchsorted(cum[cur], u[i]))
        out[i] = cur
    return out


def _near_tie(loss: float) -> bool:
    """True when a pre-rounding loss sits within TIE_MARGIN of a 4-dp rounding
    boundary — where equally-correct float64 implementations can legitimately
    disagree on the rounded digit (same artifact class as odt's fix)."""
    return abs(((loss * 1e4) % 1.0) - 0.5) * 1e-4 < TIE_MARGIN


def _world(seed: int, attempt: int):
    """One candidate world from a seed-derived sub-seed (deterministic
    resampling — the gate loop walks `attempt` like odt's _detie nudges)."""
    rng = np.random.default_rng([seed, attempt])
    n_vocab = len(VOCAB)
    base_trans = rng.dirichlet(np.full(n_vocab, 0.5), size=n_vocab)
    other = rng.dirichlet(np.full(n_vocab, 0.5), size=n_vocab)
    shifted = 0.3 * base_trans + 0.7 * other  # the SHIFTED distribution
    shifted = shifted / shifted.sum(axis=1, keepdims=True)
    base_idx = _sample_chain(rng, base_trans, N_BASE)
    ft_idx = _sample_chain(rng, shifted, N_FINETUNE)
    eval_idx = _sample_chain(rng, shifted, N_EVAL)
    # Deterministic "pre-training": Laplace-smoothed bigram logits of the base
    # corpus (softmax(logits) == the smoothed bigram probabilities).
    base_logits = np.log(_bigram_counts(base_idx) + 1.0)
    return base_logits, ft_idx, eval_idx


def generate(seed: int, out: Path) -> dict:
    sfx = instance_suffix(seed)
    job_name = JOB_BASE + sfx
    model_name = MODEL_BASE + sfx

    # --- world + analytic gates (deterministic resample until clean) ----------
    for attempt in range(MAX_RESAMPLES):
        base_logits, ft_idx, eval_idx = _world(seed, attempt)
        ft_counts = _bigram_counts(ft_idx)
        eval_counts = _bigram_counts(eval_idx)
        base_loss = _cross_entropy(base_logits, eval_counts)
        truth_loss = _cross_entropy(_finetune(base_logits, ft_counts, ITERATIONS), eval_counts)
        under_loss = _cross_entropy(
            _finetune(base_logits, ft_counts, ITERATIONS // 10), eval_counts
        )
        if base_loss - truth_loss < MIN_IMPROVEMENT:
            continue  # fine-tuning must measurably help
        if any(_near_tie(x) for x in (truth_loss, base_loss, under_loss)):
            continue  # pre-rounding loss too close to a 4-dp boundary
        break
    else:
        raise GateError(f"no clean world within {MAX_RESAMPLES} resamples (seed={seed})")

    metrics = {"eval_loss": round(truth_loss, 4), "base_eval_loss": round(base_loss, 4)}
    variant_metrics = {
        "no_finetune": {"eval_loss": round(base_loss, 4), "base_eval_loss": round(base_loss, 4)},
        "undertrained": {"eval_loss": round(under_loss, 4), "base_eval_loss": round(base_loss, 4)},
    }
    for name, vm in variant_metrics.items():
        if vm["eval_loss"] == metrics["eval_loss"]:
            raise GateError(f"variant {name!r} matches truth (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    np.savez(out / "data" / "base_model.npz", logits=base_logits, vocab=np.array(VOCAB))
    (out / "data" / "finetune.txt").write_text("".join(VOCAB[i] for i in ft_idx))
    (out / "data" / "eval.txt").write_text("".join(VOCAB[i] for i in eval_idx))
    (out / "data" / "finetune_model.py").write_text(FINETUNE_SCRIPT)

    # Gate: the EMITTED script, run from the emitted data files in a clean temp
    # dir, must reproduce the truth metrics EXACTLY.
    with tempfile.TemporaryDirectory(prefix="mlpab-finetune-gate-") as td:
        for f in ("base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"):
            shutil.copy(out / "data" / f, Path(td) / f)
        proc = subprocess.run(
            [sys.executable, "finetune_model.py"],
            cwd=td,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise GateError(f"emitted finetune_model.py failed (seed={seed}): {proc.stderr[-500:]}")
        got = json.loads((Path(td) / "metrics.json").read_text())
        if got != metrics:
            raise GateError(
                f"emitted finetune_model.py does not reproduce the "
                f"truth metrics (seed={seed}): {got} != {metrics}"
            )

    (out / "prompt.txt").write_text(
        "The directory data/ contains a base character-level language model "
        "checkpoint (data/base_model.npz: bigram logits and vocab), a "
        "fine-tuning corpus (data/finetune.txt), a held-out evaluation set "
        "(data/eval.txt) and a provided, fully deterministic fine-tuning "
        "script (data/finetune_model.py) that reads all three from its "
        "working directory and writes finetuned_model.npz plus metrics.json "
        '({"eval_loss": ..., "base_eval_loss": ...}). Do not modify the '
        "script or its hyperparameters; run it as-is.\n"
        f"Run the fine-tune AS A JOB ON THE PLATFORM — named `{job_name}` — "
        "not locally on this machine.\n"
        "Then register the fine-tuned model in the platform's model registry "
        f"as `{model_name}`, version {VERSION}, attaching the metrics from "
        "the job's metrics.json to the registry entry, with the fine-tuned "
        "model file as the model's content.\n"
        "Finally write submission/answers.json as:\n"
        f'    {{"job_name": "{job_name}", "model_name": "{model_name}", '
        '"eval_loss": <metrics.json eval_loss>, '
        '"base_eval_loss": <metrics.json base_eval_loss>}\n'
    )
    truth = {
        "family": "llm_finetuning",
        "seed": seed,
        "job_name": job_name,
        "model_name": model_name,
        "version": VERSION,
        "metrics": metrics,
        "variant_metrics": variant_metrics,
        "variant_diagnosis": VARIANT_DIAGNOSIS,
    }
    (out / "solution" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "instance.json").write_text(
        json.dumps({"family": "llm_finetuning", "seed": seed}, indent=2)
    )

    # --- gates (call the grade fn directly — no platform needed) ----------------
    reference = {"job_name": job_name, "model_name": model_name, **metrics}
    if not grade_answers(out, "none", reference)["success"]:
        raise GateError(f"reference answers fail the grade fn (seed={seed})")
    no_ft = {**reference, "eval_loss": metrics["base_eval_loss"]}
    rep = grade_answers(out, "none", no_ft)
    if rep["success"] or rep.get("diagnostic") != VARIANT_DIAGNOSIS["no_finetune"]:
        raise GateError(
            f"no_finetune answers not rejected with their diagnosis (seed={seed}): {rep}"
        )
    under = {**reference, "eval_loss": variant_metrics["undertrained"]["eval_loss"]}
    rep = grade_answers(out, "none", under)
    if rep["success"] or rep.get("diagnostic") != VARIANT_DIAGNOSIS["undertrained"]:
        raise GateError(
            f"undertrained answers not rejected with their diagnosis (seed={seed}): {rep}"
        )
    wrong_name = {**reference, "model_name": f"{model_name}final"}
    if grade_answers(out, "none", wrong_name)["success"]:
        raise GateError(f"wrong-name answers pass the grade fn (seed={seed})")

    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-llm-finetuning-selftest/{seed}"))
            print(
                f"[llm_finetuning] seed={seed} metrics={meta['metrics']} gates=OK "
                "(incl. subprocess reproduction)"
            )
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
