---
name: hops-opt-session
description: This CLI caches project/feature-store ids and resolves your own store first, so repeated `hops` calls are cheap. Auto-invoke when issuing multiple feature-group/view CLI calls.
---

# opt2: session reuse is automatic

This CLI build persists the project / feature-store id and checks **your own
store first** when resolving a feature group. Repeated `hops fg ...` / `hops fv ...`
calls reuse that cached session instead of re-resolving and re-authenticating.

Practical effect: **do not contort the workflow to minimize call count** for
session cost — each call after the first is cheap. Just issue the natural
inspect-before-write reads (`hops fg list`, `info`, `preview`) freely; the cache
absorbs the cost. Reference FGs by plain `name[:version]` — own-store resolution
is the fast default.
