# Ideas ledger

A durable, greppable record of zone-based transformer ideas — shipped, in
progress, or proposed — so nothing raised in a conversation gets lost, and
so "is this already built?" has one place to check before starting new
work. Several of these trace back to the original handwritten notebook
pages behind zoneboost, transcribed in [`docs/origin-story.html`](docs/origin-story.html).

New ideas get appended as a new row with status `Proposed`, whatever detail
is known at the time, and a one-line source note. When an idea ships,
update its row's status and "where it lives" column rather than deleting
it — the history of what an idea used to look like before it shipped is
useful context on its own.

| Idea | Status | Source | Where it lives / would hook in |
|---|---|---|---|
| Pairwise zone-grid weak learner | Shipped | Notebook page 1 | `src/zoneboost/_weak_learner.py` |
| Empirical Bayes shrinkage | Shipped | Notebook pages 2-3 | `src/zoneboost/_shrinkage.py` |
| Honest data splits / cross-fitted cell means | Shipped | Notebook page 3 | `src/zoneboost/regressor.py` |
| Bootstrap stability | Shipped | Notebook page 3 | `src/zoneboost/_bootstrap.py` |
| Mondrian conformal prediction | Shipped | — | `src/zoneboost/regressor.py` (`mondrian_col`) |
| Time-based drift comparison (`compare_models`) | Shipped, no alerting yet | Notebook page 2 | `src/zoneboost/_drift.py` |
| **ZoneProfileEncoder** | **Shipped** | Notebook pages 2-3 | `src/zoneboost/_zone_profile.py` |
| **DepthTransformer** | **Shipped** | Notebook page 3 (core/outlier rings) | `src/zoneboost/_depth.py` |
| **ConditionalZoneGrid** | **Shipped** | Notebook page 1 (`a=1 & b=1` filtering) | `src/zoneboost/_conditional_grid.py` |
| **Drift threshold/alert monitor** | **Shipped** | Notebook page 2 (red-ink date note) | `src/zoneboost/_drift_alert.py` (`flag_drift`) |
| **LLM zone auto-naming (business language)** | **Shipped** | Strategy discussion, not the notebook | `src/zoneboost/_zone_namer.py` (`LLMZoneNamer`), gated behind the `zoneboost[llm]` extra |

## Detail

**Pairwise zone-grid weak learner.** Split every predictor into adaptive
zones, average the current residual per zone (or zone pair), boost the
result. The core mechanism; see `docs/how-it-works.html`.

**Empirical Bayes shrinkage.** A DerSimonian-Laird method-of-moments
estimate of shrinkage strength (`_estimate_shrinkage_m`), so a sparse
zone's own mean leans toward its hierarchical prior instead of overfitting
a handful of rows.

**Honest data splits / cross-fitted cell means.** Never grade a zone's
own contribution on the same rows that defined it — the boosting
estimator's cross-fitting discipline.

**Bootstrap stability.** Refit the whole model on resampled data
(`n_bootstrap` full refits) to report genuine across-refit variance in a
contribution, an importance, or a prediction — distinct from a single
fit's own reliability diagnostics.

**Mondrian conformal prediction.** Per-group calibration
(`mondrian_col`/`mondrian_min_group_size` on `ZoneBoostRegressor`) so a
minority segment gets its own conformal margin instead of one pooled
margin dominated by the majority segment.

**Time-based drift comparison (`compare_models`).** A stateless diff
between two already-fitted models (e.g. last quarter's vs. this quarter's):
feature-importance change, boundary/population shift, prediction shift.
Purely descriptive — no threshold, no alert, no state retained across
calls. See "Drift threshold/alert monitor" below for the gap.

**ZoneProfileEncoder.** `sklearn`-compatible `TransformerMixin` that fits
the same per-column zone construction the core estimator uses, then emits
each zone's (empirical-Bayes-shrunk) mean, variance, and support count as
new feature columns — usable ahead of *any* downstream model, not only
zoneboost's own estimators. See `src/zoneboost/_zone_profile.py`,
`README.md` ("Zone profile encoding"), `docs/how-it-works.html`
(`#zone-profile-encoder`), `docs/api-reference.html`
(`#zone-profile-parameters`).

**DepthTransformer.** Generalizes the notebook's discrete
inner-core/outer-core/outlier rings into a continuous "coreness" score
over a group of numeric columns, via **Mahalanobis distance** — a point's
distance from the joint mean of the group, scaled by their covariance.
Tukey halfspace depth and convex-hull peeling were considered and
rejected: halfspace depth has no simple closed form past ~2 dimensions,
and convex-hull peeling needs `scipy.spatial.ConvexHull`, a dependency
this package doesn't otherwise carry. Emits both the raw distance and a
bounded `1 / (1 + distance)` rescaling (disclosed as a monotonic
rescaling, not a calibrated percentile), with `np.linalg.pinv` + a ridge
term guarding against singular/ill-conditioned covariance. No discrete
region labels — deliberately deferred, since a continuous score composes
with any downstream model. See `src/zoneboost/_depth.py`, `README.md`
("Depth transformer"), `docs/how-it-works.html` (`#depth-transformer`),
`docs/api-reference.html` (`#depth-parameters`).

**ConditionalZoneGrid.** Fits a 2D zone grid over two continuous columns
*separately within each discrete segment* (the notebook's "keep
filtering: (x,y) if a=1 & b=1 & c=1..."). Built as a standalone
transformer, the same `ZoneProfileEncoder`/`DepthTransformer` sibling
pattern, not folded into the boosting/`explain()` machinery — a real
scope decision, since the original framing here assumed sequencing after
functional-ANOVA purification (`_purify.py`) to avoid double-counting
attribution. Read in full: purification only rewrites `explain(X)`'s
already-computed contribution columns and has nothing to do with a
standalone transformer that never touches `rounds_`/`explain()`, so that
concern doesn't apply to what actually shipped. A segment below
`min_segment_size` (or unseen at `fit` time) falls back to a single
pooled global grid, with a `"..._used_segment_grid"` flag disclosing
which grid a row actually got. See
`src/zoneboost/_conditional_grid.py`, `README.md` ("Conditional zone
grids"), `docs/how-it-works.html` (`#conditional-zone-grid`),
`docs/api-reference.html` (`#conditional-grid-parameters`). If a
boosting-integrated version (feeding `predict`/`explain` directly, a true
alternative to the 3-way interaction search) is ever wanted instead,
that's a materially larger, different change to `_weak_learner.py` itself
— not covered by what shipped here.

**Drift threshold/alert monitor.** `flag_drift(model_old, model_new,
X_eval, y_eval=None, alpha=0.1)` turns `compare_models`'s stateless diff
into an active alert: flags when the observed prediction shift between
two model snapshots exceeds `model_new`'s own already-calibrated
split-conformal margin (the same quantity `predict_interval` uses), and
when `mondrian_col` was set, additionally flags any per-group shift that
exceeds that group's own margin from `conformal_scores_by_group_` —
reusing Mondrian conformal prediction's existing per-group calibration
scores rather than inventing new calibration machinery. Ships as a new
function in a new file, `src/zoneboost/_drift_alert.py`, which *calls*
`compare_models` rather than editing it in place — `_drift.py` itself has
a zero-line diff. Disclosed as a heuristic significance check, not a
formal hypothesis test. See `README.md` ("Drift threshold/alert monitor"
under "Time-based drift comparison"), `docs/explaining-predictions.html`
(`#drift-threshold-monitor`), `docs/api-reference.html`
(`#flag-drift-signature`).

**LLM zone auto-naming.** `LLMZoneNamer.name_zones(zone_summaries,
context=None)` turns a batch of plain zone-description dicts into short
business-language names ("young, low-affordability, high-claims
corridor") via the Claude API — so an audit artifact reads like an
underwriting manual instead of a table of cut points. Asked the user for
the scope decision this item was flagged as needing; they chose "separate
optional extra": ships inside the `zoneboost` package
(`src/zoneboost/_zone_namer.py`), gated behind `pip install
zoneboost[llm]` (the new `anthropic` optional dependency in
`pyproject.toml`), and never imported eagerly — `anthropic` is only
imported inside `LLMZoneNamer`'s own method bodies, so `import zoneboost`
(and even `from zoneboost import LLMZoneNamer`) keeps working with zero
extra dependencies installed; verified directly by blocking the
`anthropic` import and confirming both still succeed. `client` is
injectable (any object exposing `.messages.create(...)`), which is what
makes the test suite fully offline — no network call, no API key, no
`anthropic` runtime dependency required to run `pytest`. Decoupled from
every other zoneboost internal on purpose: `zone_summaries` is a plain
list of dicts the caller builds from `ZoneProfileEncoder.zone_stats_`,
`ConditionalZoneGrid.segment_grids_`, or by hand, not something this class
parses out of `rounds_` itself. See `README.md` ("LLM zone naming
(optional)"), `docs/how-it-works.html` (`#llm-zone-naming`),
`docs/api-reference.html` (`#llm-zone-namer-parameters`).
