# banter

A testbed that drives Claude Code against vendor CLIs, MCP servers, and
Python SDKs on top of [MLE-bench](https://github.com/openai/mle-bench).
Two loops sit on top of the engineer-runs-one-challenge primitive:

- **benchmark** — run a fixed list of (challenge × interface × skills)
  once each and compare.
- **autoresearch** — let a researcher Claude *iteratively edit the
  interface source* across versions, re-running the engineer after each
  edit, until the engineer's metrics improve.

Supporting work for the KTH master's thesis in `thesis/`
(`thesis/thesis.tex`). The thesis answers three research questions:

- **RQ1** — Can an LLM-driven research loop improve a *CLI* interface
  measurably (score / cost / tool-call mix) over committed baseline?
- **RQ2** — Same question for an *MCP server* interface.
- **RQ3** — Does providing per-interface *skills* / docs further shift
  the result, and how does the answer differ across CLI / MCP / SDK?

The testbed validates these RQs with a *fake* SaaS platform (`mlkit`,
ships in-repo) and the real `hopsworks` platform.

```
banter/
  testbed/   ← framework, configs, results — everything below assumes `cd testbed`
  thesis/    ← KTH LaTeX thesis (thesis/thesis.tex)
```

---

## Install

```bash
cd testbed
make install        # creates .venv, installs banter + deps, libomp, etc.
banter setup interfaces/mlkit/sdk/config.yaml   # optional — writes any API keys
```

`make install` is the only install target. macOS only for now (uses APFS
clones + Keychain for the Claude OAuth token).

---

## Run the mlkit example

`mlkit` is a fake AutoML platform (Python + tiny local HTTP server) that
ships with the repo in three shapes — CLI, SDK, MCP server. The engineer
solves MLE-bench's `aerial-cactus-identification` using whichever shape
is configured.

```bash
# one-shot benchmark — engineer runs the mlkit sdk once
banter interfaces/mlkit/benchmark/sdk/config.yaml

# autoresearch — researcher iterates the mlkit sdk source across v0..v3
banter interfaces/mlkit/autoresearch/sdk/config.yaml
```

Swap `sdk` for `cli` or `mcp` for the other two shapes. Each run
produces a folder under `results/benchmark/` or `results/autoresearch/`
with the engineer's transcript, `submission.csv`, MLE-bench grading,
and a `results.csv` summary.

---

## Add your own interface

An interface is a directory under `testbed/interfaces/<name>/<type>/`
where `<type> ∈ {cli, mcp, sdk}`. It needs:

- a buildable client (wheel, npm package, binary — anything `install:`
  can produce) plus its source,
- a `config.yaml` describing how to build, auth, test, and run it.

### Interface `config.yaml`

```yaml
# interfaces/myapi/sdk/config.yaml
install:                           # build → produces a wheel/binary in $INTERFACE_DIR
  - pip wheel . --no-deps --no-build-isolation -w $INTERFACE_DIR

auth_command: python -c "import myapi; myapi.login()"
test_command: python -c "import myapi; assert myapi.__version__"

binary: myapi-0.1.0-py3-none-any.whl
runtime_install:                   # installed into each per-challenge venv
  - pip install --no-deps $INTERFACE_DIR/myapi-0.1.0-py3-none-any.whl

keys:                              # surfaced to engineer + researcher env
  MYAPI_API_KEY: ""

allowed_domains:                   # network allowlist (sandbox)
  - api.myapi.example

prompt: |
  An `myapi` Python SDK is installed. Use it to solve the competition:
    import myapi; myapi.login()
    myapi.fit("data")
    myapi.predict(model, "data", "submission/submission.csv")
```

### Benchmark config that uses it

```yaml
# interfaces/myapi/benchmark/sdk/config.yaml
engineer_model: claude-sonnet-4-6
max_seconds: 3600
tasks:
  image_classification: [aerial-cactus-identification]
interfaces:
  - {config: interfaces/myapi/sdk/config.yaml}
skills: [none]
```

Run with `banter interfaces/myapi/benchmark/sdk/config.yaml`.

### Autoresearch config that improves it

```yaml
# interfaces/myapi/autoresearch/sdk/config.yaml
engineer_model: claude-sonnet-4-6
researcher_model: claude-opus-4-7

tasks:
  image_classification: [aerial-cactus-identification]

interfaces:
  - config: interfaces/myapi/sdk/config.yaml

skills: none
docs: https://github.com/myorg/myapi-docs.git    # optional — cloned per run
improve:
  - interface                                    # researcher may edit the SDK source

goals:                                           # optimisation targets
  - {metric: score,         direction: maximize}
  - {metric: total_tokens,  direction: minimize}
  - {metric: wall_time_s,   direction: minimize}
  - {metric: sdk_calls,     direction: maximize}
  - {metric: python_calls,  direction: minimize}

budget:
  max_increments: 3                              # produces v0 + v1..v3
  max_cost_usd: .inf
```

Run with `banter interfaces/myapi/autoresearch/sdk/config.yaml`. The
researcher will produce `v0..v3`, each a full engineer run with edits
to the interface source, and write a per-version row to `results.csv`.

---

## Skills

A skill is a Claude Code slash command bundled with an interface. Drop
a `SKILL.md` at:

```
interfaces/<name>/skills/<skill-name>/SKILL.md
```

…and reference it from a benchmark / autoresearch config:

```yaml
skills:
  - <skill-name>          # or `none`
```

The engineer's prompt gets the available skills appended; the
researcher can also edit them when `improve:` includes `skills`.

---

## Docs

Reference documentation that should be visible to both researcher and
engineer is set per autoresearch config:

```yaml
docs: https://github.com/myorg/myapi-docs.git   # git URL (shallow-cloned per run)
# docs: /Users/me/local/docs                    # or a local path
# docs: none                                    # default — no docs bundle
```

A copy lands at `<run>/docs/` and `<challenge>/docs/`. The engineer and
researcher see a `Reference docs` section in their prompts pointing at
those paths.

---

## Results

```
testbed/results/benchmark/      ← one folder per benchmark run
testbed/results/autoresearch/   ← one folder per autoresearch run; v0..vN inside
testbed/results/*/results.csv   ← rollup, one row per challenge (or version)
```

Each run dir contains the engineer's full stream-json transcript,
`submission.csv`, MLE-bench's `grading.json`, and a per-tool-call
`commands.jsonl` — enough to reproduce any number in the rollup.

---

## Tests

```bash
make test               # 106 unit tests
make test-integration   # 2 live sandbox / OAuth integration tests
```
