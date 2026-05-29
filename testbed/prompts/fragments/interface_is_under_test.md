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
