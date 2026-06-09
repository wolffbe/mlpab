**Optimize ALL goals JOINTLY — they are NOT ranked.** The order above is not a
priority order. Your objective is the **composite** (`normalized_composite`'s J,
an equal-weight blend of every goal) — a **Pareto-optimal** result across all
goals, not one goal maximized and then the next. Do NOT treat `score` (or any
single metric) as primary and the others as optional tie-breakers. Each version,
push the goal with the **lowest** normalized contribution, and prefer changes
that raise a lagging goal **without regressing** the others (a Pareto improvement).
"The task didn't need endpoint X" / "that metric isn't required to solve this
challenge" is **NOT** a valid reason to leave a goal like `whitelist_hits` low —
it is an explicit optimization target, so extend the interface until the engineer
exercises it. A version that maxes one goal while another sits at the floor is a
BAD result, even if that one goal looks great.
