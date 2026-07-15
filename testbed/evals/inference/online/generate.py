"""Online-serving task (FTI sub-category: inference/online) — generator.

Usage:
    python -m evals.inference.online.generate --seed 7 --out /tmp/online-7
    python -m evals.inference.online.generate --selftest

World: account feature profiles (`data/features.csv` — account_id unique key,
f1..f4) and a lookup list (`data/lookup_keys.txt` — 20 of the account_ids,
one per line). The agent must load the profiles into a feature table
`profiles<sfx>` (per-instance suffix) with ONLINE/low-latency access enabled,
then retrieve each lookup
key's feature vector THROUGH the online/low-latency read path and write
`submission/answers.json`:

    {"vectors": {"<account_id>": [f1, f2, f3, f4], ...}}

Ground truth by construction (the generator made the rows). Generation-time
gates run the grade function (adapter `none`): the reference vectors pass; a
corrupted vector and a missing key both fail.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evals.common import instance_suffix

KIND = "platform"  # deliverable kind: table | dataset | answers | platform
SUMMARY = "Measures whether an agent can materialize features for online serving and retrieve them through the platform's low-latency read path, not by re-reading the source data."

N_ACCOUNTS = 120
N_LOOKUPS = 20
FEATURES = ["f1", "f2", "f3", "f4"]
TABLE_BASE = "profiles"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    table = TABLE_BASE + instance_suffix(seed)
    df = pd.DataFrame({"account_id": [f"A{i:04d}" for i in range(N_ACCOUNTS)]})
    for f in FEATURES:
        df[f] = np.round(rng.normal(rng.uniform(-5, 5), rng.uniform(0.5, 3), N_ACCOUNTS), 4)
    keys = sorted(rng.choice(df["account_id"], N_LOOKUPS, replace=False).tolist())
    vectors = {k: [float(df.loc[df["account_id"] == k, f].iloc[0]) for f in FEATURES] for k in keys}

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    df.to_csv(out / "data" / "features.csv", index=False)
    (out / "data" / "lookup_keys.txt").write_text("\n".join(keys) + "\n")
    (out / "data" / "schema.md").write_text(
        "# Schema\n\n"
        "- **features.csv**: account_id (unique key), "
        + ", ".join(FEATURES)
        + " (numeric features)\n"
        "- **lookup_keys.txt**: one account_id per line — the keys to serve\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains account feature profiles "
        "(data/features.csv: account_id, "
        f"{', '.join(FEATURES)}) and a lookup list (data/lookup_keys.txt — one "
        "account_id per line).\n"
        f"Load the profiles into a feature table named `{table}`, version 1, on "
        "the platform, with record key `account_id`, and enable ONLINE / "
        "low-latency access for it.\n"
        "Then retrieve the feature vector for EACH key in data/lookup_keys.txt "
        "THROUGH the online/low-latency read path (not by re-reading the CSV or "
        "the offline store), and write submission/answers.json as:\n"
        '    {"vectors": {"<account_id>": [f1, f2, f3, f4], ...}}\n'
        "with the feature values as loaded (floats, in the order "
        f"{', '.join(FEATURES)}), one entry per lookup key.\n"
    )
    truth = {
        "family": "online",
        "seed": seed,
        "table_name": table,
        "table_version": 1,
        "features": FEATURES,
        "keys": keys,
        "vectors": vectors,
        "record_ids": df["account_id"].tolist(),
    }
    (out / "solution" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "online", "seed": seed}, indent=2))

    # --- gates: the grade function accepts truth, rejects corruptions ----------
    from evals.inference.online.grade import grade

    def run(answers: dict) -> bool:
        with tempfile.TemporaryDirectory(prefix="mlpab-online-gate-") as td:
            run_dir = Path(td)
            (run_dir / "submission").mkdir()
            (run_dir / "submission" / "answers.json").write_text(json.dumps(answers))
            return grade(out, "none", run_dir)["success"]

    if not run({"vectors": vectors}):
        raise GateError(f"reference vectors fail the grade function (seed={seed})")
    corrupted = {k: list(v) for k, v in vectors.items()}
    corrupted[keys[0]][0] += 0.001
    if run({"vectors": corrupted}):
        raise GateError(f"corrupted vector passes the grade function (seed={seed})")
    partial = {k: vectors[k] for k in keys[1:]}
    if run({"vectors": partial}):
        raise GateError(f"missing lookup key passes the grade function (seed={seed})")
    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-online-selftest/{seed}"))
            print(f"[online] seed={seed} keys={len(meta['keys'])} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
