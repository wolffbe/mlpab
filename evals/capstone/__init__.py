"""Capstone FTI evals — end-to-end modelling tasks (regression / classification).

Unlike the deterministic single-pillar tasks (whose grader checks a content
digest), a capstone exercises the WHOLE feature→training→inference stack and is
graded HYBRID: the held-out predictive metric must clear a calibrated bar AND
the on-platform FTI artifacts (feature group, training dataset, registered
model) must exist (verified through the adapter's state reads). See
`evals/capstone/common.py` for the shared scaffolding.

The raw data is built ONCE, offline, by the `_data/build_*.py` scripts (which
may need an API key) and committed as a CSV fixture. At run time the seeded
`generate()` only reads that committed CSV — no network, no key.
"""
