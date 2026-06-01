You are solving the MLE-bench competition `{challenge_id}`.

The competition data is available in the `data/` directory (a symlink to the
prepared MLE-bench dataset). Read the competition description and any provided
files there to understand the task, the target, and the required submission
format.

Your goal is to produce a valid `submission/submission.csv` against the
competition's metric in a single attempt. Concretely:

1. Explore `data/` to understand the schema, the target, and the sample
   submission format.
2. Build and train a model (you may install whatever Python libraries you need
   into the current environment).
3. Generate predictions for the test set and write them to
   `submission/submission.csv` in exactly the format the competition expects.

{fragment}

<!--UNDER_TEST_START-->
## The interface is what's being measured — do NOT route around it

The interface above is the unit under test. The researcher is iterating its
source to improve it; your job is to **USE the interface as-is** and report
what it produces. The score the interface earns is the data the researcher
needs.

- A trivial or low-quality prediction from the interface is **NOT a bug**.
  All-zeros, all-0.5, all-NaN, low-AUC, etc. from the interface are valid
  experimental results. Submit them.
- **DO NOT write your own training script** when the interface produces a
  poor result. That defeats the experiment — the researcher would see a
  great score that has nothing to do with the interface.
- A "bug" that justifies redoing work means YOUR driver code crashed
  (Python exception, failed install, malformed submission file). It does
  NOT mean "the interface returned predictions I don't like."
- If the interface is genuinely broken (raises, can't be called, returns
  wrong shape), still produce SOMETHING that grades — e.g. copy
  `data/sample_submission.csv` to `submission/submission.csv` so the
  competition can score it. That's the floor result; the researcher will
  see it and know the interface is broken at this version.

Faithfully exercise the interface and ship its output as the submission.
Do not try to outperform it.
<!--UNDER_TEST_END-->

Environment and constraints (detected on this machine):
{environment}
- Run training in the FOREGROUND and let it finish; never launch background or
  daemon processes that could outlive your command. Do not call `Bash` with
  `run_in_background: true` — chain commands synchronously instead.
- Keep the model small and training short (a few epochs) so your single attempt
  completes well within the time budget. Speed matters more than squeezing out
  the last bit of accuracy.

## Sandbox boundary — read before troubleshooting any "permission denied"

You run with a tightly-scoped sandbox. The ONLY restriction that actually
limits you is on READS — write freely inside your cwd.

- **Your boundary = your current working directory.** Read + write freely
  inside it. `./submission/submission.csv`, `./model.pt`, `./.hf/...`,
  `./.tmp/...`, `./venv/...` — all reachable. Use relative paths.
- **Shared infrastructure reachable through indirection**: `./venv/` resolves
  to the shared Python libs (via .pth), `./data/` is a symlink to the
  mle-bench cache. You don't address these by absolute path; just use the
  local entry points.
- **Everything else is invisible** — other challenges, other versions, the
  researcher's prompts, `~/.ssh`, `~/.aws`, anything under `/Users/<you>/`
  outside your boundary. Don't try to read them.
- **Network**: unrestricted (pip, Hugging Face, Kaggle, GitHub all OK).
- **Denied tools**: `Task`, `Agent`, `WebFetch`, `WebSearch`, `Cron*`,
  `Schedule*`, `EnterPlanMode`, `EnterWorktree`. Don't call them.

**Bash tool flags that are HARD-BLOCKED by a PreToolUse hook** (the call is
rejected before it runs — do not use them):
- `dangerouslyDisableSandbox: true`
- `run_in_background: true`

For tools that try to write to `/tmp` or `~/.cache` (HF, Torch, Triton), point
them at cwd via env vars BEFORE invoking, not after a failure:
```bash
HF_HOME=$PWD/.hf TORCH_HOME=$PWD/.torch TMPDIR=$PWD/.tmp \
  python train_model.py
```

**STOP reasoning about sandbox internals.** Do NOT speculate about
denyWithinAllow rules, `.git/HEAD/objects/refs` denies, `"."` not expanding
to cwd, profile languages, or any other sandboxing minutiae — none of it
applies here.

If `open('submission/submission.csv', 'w')` fails for any reason, the recipe
is plain Python — not sandbox debugging:
```python
import os, pathlib
print("CWD =", os.getcwd())              # 1. confirm cwd
pathlib.Path("submission").mkdir(exist_ok=True)  # 2. ensure dir
with open("submission/submission.csv", "w") as f: ...  # 3. relative write
```

Make a SINGLE attempt at the challenge. Train one model end to end and write
`submission/submission.csv` exactly once — do not iterate over multiple models,
tune hyperparameters, or regenerate the submission to chase a higher score.
Whatever that one attempt scores is the result; a mediocre score is acceptable.
The only reason to redo work is an error — a bug, a crash, a failed install — in
which case fix it and continue. Once a valid `submission/submission.csv` exists,
you are done.

Work autonomously end to end. When you are done, the file
`submission/submission.csv` must exist and be valid for grading.
