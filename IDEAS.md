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
| ConditionalZoneGrid | Proposed | Notebook page 1 (`a=1 & b=1` filtering) | sequence after `_purify.py`, since attribution should be canonicalized before nesting grids |
| Drift threshold/alert monitor | Proposed | Notebook page 2 (red-ink date note) | extends `_drift.py`'s `compare_models`, reuses Mondrian's per-group calibration scores |
| LLM zone auto-naming (business language) | Proposed, needs a scope decision | Strategy discussion, not the notebook | would sit outside the core package — flag before building; zoneboost is currently a zero-ML-dependency, numpy/pandas-only library, and this is the first idea that would break that |

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

**ConditionalZoneGrid** (proposed). Fit separate zone grids per discrete
segment (the notebook's "keep filtering: (x,y) if a=1 & b=1 & c=1..."),
capturing third-order-and-up interactions while keeping each segment's own
grid visual and auditable. Natural sequencing point: after the
functional-ANOVA purification work (`src/zoneboost/_purify.py`), since
attribution should be canonicalized before nesting grids inside segments —
otherwise a segment's own grid and the top-level grid could double-count
the same effect.

**Drift threshold/alert monitor** (proposed). Extends `compare_models`
(`_drift.py`) from a stateless diff into an active flag: alert when a
zone's conditional mean has shifted beyond a conformal band, rather than
requiring a person to eyeball the diff. Reuses the per-group calibration
scores Mondrian conformal prediction already computes
(`regressor.py`'s `mondrian_col`) rather than inventing new calibration
machinery.

**LLM zone auto-naming** (proposed, stretch). Auto-name a zone or
zone-pair in business language ("young, low-affordability, high-claims
corridor") so an audit artifact reads like an underwriting manual instead
of a table of cut points. Needs an explicit scope decision before building:
zoneboost is currently a zero-ML-dependency, numpy/pandas-only library, and
this is the first idea that would introduce an external (LLM) dependency —
likely belongs in a separate optional extra or a docs/tooling layer, not
the core package, if built at all.
