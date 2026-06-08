You are solving the MLE-bench competition `{challenge_id}`.

The competition data is available in the `data/` directory (a symlink to the
prepared MLE-bench dataset). Read the competition description and any provided
files there to understand the task, the target, and the required submission
format.

Your goal is to produce a valid `submission/submission.csv` against the
competition's metric in a single attempt.

{fragment}

<!--LOCAL_ONLY_START-->
Concretely:

1. Explore `data/` to understand the schema, the target, and the sample
   submission format.
2. Build and train a model (you may install whatever Python libraries you need
   into the current environment).
3. Generate predictions for the test set and write them to
   `submission/submission.csv` in exactly the format the competition expects.
<!--LOCAL_ONLY_END-->

<!--UNDER_TEST_START-->
## Restrictions — everything runs on the platform, nothing runs locally

All real work — feature engineering, model training, inference — MUST run on the
PLATFORM, driven through the interface described above (e.g. as a remote job).
**Nothing is computed on this machine.** The interface and every dependency it
needs are **already installed** — you never install anything yourself.

**These rules are HARD-ENFORCED.** A violating command is rejected before it runs
(you'll get a `DENIED: …` message explaining why). Don't fight it or look for a
way around it — adapt, or give up cleanly (below).

**You may ONLY use the interface under test** (exactly one of these applies to
this run — the section above tells you which):
- **CLI mode** → only the platform's CLI command. No local Python at all.
- **MCP mode** → only the `mcp__…__*` tools. No local Python, no CLI.
- **SDK mode** → only the platform's Python SDK (`import` it to drive the
  platform — not to compute locally). No CLI, no MCP tools.

Bash is a **fail-closed allowlist**: only the interface under test plus basic
shell utilities run — *every other executable is denied by default*. Don't reach
for a different tool when the interface can't do something; that's your signal to
give up cleanly (below).

**Always BLOCKED (every mode):**
- Local model training / ML libraries (`torch`, `tensorflow`, `keras`,
  `sklearn`, `xgboost`, `lightgbm`, …) — training must be a remote job.
- Any interface other than the one under test.
- **Any general-purpose interpreter or other binary** — `node`, `ruby`, `perl`,
  `php`, `deno`, `Rscript`, … — and **network tools** (`curl`, `wget`). There is
  no local-scripting escape hatch; if the interface can't do it, nothing can.
- In CLI and MCP mode: **all** local Python, including `pip` / `python -m pip`
  (there is nothing to install — it's done for you).

**Allowed in every mode (this is the whole list):**
- The interface under test (CLI command / MCP tools / SDK `import`, per above).
- Basic shell only — inspecting the task (`cat`, `head`, `ls`, `grep`, `wc`, …
  and the `Read` tool on `data/`) and writing the submission (`mkdir`, `cp`).

## One attempt — then give up cleanly

Make **one** attempt with the interface exactly as it is. You are a probe: the
researcher reads your result and improves the interface for the next run.

- If the interface **cannot** run the work remotely (the capability is missing,
  a command/tool doesn't exist, it errors), **do NOT build a workaround and do
  NOT debug at length.** STOP, write the floor submission (below), and in your
  final message state plainly **what capability the interface is missing**. That
  report is exactly what the researcher needs.
- **You MUST always end with a graded submission.** Giving up does NOT mean
  producing nothing. If you can't get real predictions from the platform, copy
  `data/sample_submission.csv` to `submission/submission.csv` as the floor
  result — a valid, low score the researcher can act on:
  ```bash
  mkdir -p submission && cp data/sample_submission.csv submission/submission.csv
  ```

## The interface is what's being measured — do NOT route around it

The interface above is the unit under test. Your job is to **USE it as-is** and
report what it produces. The score the interface earns is the data the
researcher needs.

- A trivial or low-quality prediction from the interface is **NOT a bug**.
  All-zeros, all-0.5, low-AUC, etc. from the interface (or the floor submission)
  are valid experimental results. Submit them.
- **DO NOT train your own model** to rescue a poor interface result — and you
  can't anyway (local ML is blocked). That would defeat the experiment.
- Keep any remote job small and short (a few epochs) so your single attempt
  finishes within the time budget. Speed beats squeezing out accuracy.

Faithfully exercise the interface and ship its output as the submission.
<!--UNDER_TEST_END-->

Environment and constraints (detected on this machine):
{environment}
- Run commands in the FOREGROUND and let them finish; never launch background or
  daemon processes that could outlive your command. Do not call `Bash` with
  `run_in_background: true` — chain commands synchronously instead.
- **Never write an open-ended wait loop** such as
  `until [ -f submission/submission.csv ]; do sleep 5; done` or
  `while true; do …; done`. Run the command itself in the foreground and let
  that single command block until it returns — do NOT spawn work and then poll
  for its output. For a long command, pass the `Bash` tool a large explicit
  `timeout` so it runs to completion in one synchronous call.
- If a command is ever moved to a background task and you are handed a
  task-output file path, `Read` that exact path to inspect progress or errors —
  do not blind-wait for a notification. But keeping the command in the
  foreground (above) is what you should actually do, so this never arises.
<!--LOCAL_ONLY_START-->
- Keep the model small and training short (a few epochs) so your single attempt
  completes well within the time budget. Speed matters more than squeezing out
  the last bit of accuracy.
<!--LOCAL_ONLY_END-->

## Sandbox boundary — read before troubleshooting any "permission denied"

You run with a tightly-scoped sandbox. The ONLY restriction that actually
limits you is on READS — write freely inside your cwd.

- **Your boundary = your current working directory.** Read + write freely
  inside it. `./submission/submission.csv`, `./.tmp/...`, `./venv/...` — all
  reachable. Use relative paths.
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

<!--LOCAL_ONLY_START-->
For tools that try to write to `/tmp` or `~/.cache` (HF, Torch, Triton), point
them at cwd via env vars BEFORE invoking, not after a failure:
```bash
HF_HOME=$PWD/.hf TORCH_HOME=$PWD/.torch TMPDIR=$PWD/.tmp \
  python train_model.py
```
<!--LOCAL_ONLY_END-->

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

Make a SINGLE attempt at the challenge and write `submission/submission.csv`
exactly once — do not iterate over multiple approaches, tune, or regenerate the
submission to chase a higher score. Whatever that one attempt scores is the
result; a mediocre score is acceptable.
<!--LOCAL_ONLY_START-->
The only reason to redo work is an error — a bug, a crash, a failed install — in
which case fix it and continue.
<!--LOCAL_ONLY_END-->
Once a valid `submission/submission.csv` exists, you are done.

Work autonomously end to end. When you are done, the file
`submission/submission.csv` must exist and be valid for grading.
