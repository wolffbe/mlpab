# Re-run flag — databricks grader artifacts (fixed 2026-06-30)

7 opus combos were scored as failures by grader bugs that are now fixed in
`evals/adapters/databricks.py` + `evals/common.py`. The deliverables were
correct on the platform, but per-run schemas are torn down, so these can only
be reclaimed by RE-RUNNING the agent (re-grade alone is impossible). Do this
**after run `18` finishes** (don't mutate `results.csv` while it holds the lock).

## Fixes that make these gradeable now
- `read_rows(name, version)` probes `X_vN` / `X_N` / `X` (full_reload version-in-name).
- `get_model` reads metrics from the model version's MLflow run (capstone A5).

## Group A — full_reload (valid=False → auto-rerun)
These have no valid=True row, so `--retry` purges + re-runs them automatically:
- cli/none, cli/official, sdk/none, sdk/official  (feature/full_reload)

## Group B — capstone/ccfraud (valid=True, success=False → MUST purge first)
`--retry` treats these as "done" (they have a valid=True row) and will NOT
re-run them. Purge their CSV rows + dirs so they re-run as missing combos.
Graded-before-the-fix combos:
- cli/none, cli/official, sdk/none  (capstone/ccfraud)
- ALSO sdk/official if it finished capstone before this fix landed — check its
  grading.json for the lone A5_model_registered fail and purge it too.

## Procedure (run 18 must be stopped/finished)
```bash
cd /Users/wolffbe/workspace/banter/testbed
CFG=configs/treatments/18_db-full-cli-sdk-skills-opus.yaml

# B: purge capstone/ccfraud opus rows + dirs so they re-run as "missing"
python3 - <<'PY'
import csv, shutil, pathlib
src="results/results.csv"
rows=list(csv.DictReader(open(src)))
hdr=rows[0].keys() if rows else []
keep=[]
for r in rows:
    drop = (r['config']=='18_db-full-cli-sdk-skills-opus'
            and r['task']=='ccfraud' and r['success']!='True')
    if drop:
        d=pathlib.Path(r['run_dir'])
        if d.exists(): shutil.rmtree(d)
        print("purged", r['interface'], r['skills'])
    else:
        keep.append(r)
with open(src,'w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(hdr)); w.writeheader(); w.writerows(keep)
PY

# A + B together: --retry handles full_reload; purged capstone now runs as missing
mlpab run --skip --retry "$CFG"
```

Verify after: full_reload (4) and capstone/ccfraud A5 should pass for opus.
