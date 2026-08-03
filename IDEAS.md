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
| **Laplace history transformer** | **Shipped** | Strategy discussion, not the notebook | `src/zoneboost/_laplace_history.py` (`LaplaceHistoryTransformer`) |
| **Depth crowd (wisdom-of-the-crowd aggregation)** | **Shipped** | Strategy discussion, not the notebook | `src/zoneboost/_depth_crowd.py` (`DepthCrowd`) |
| **Categorical depth transformer** | **Shipped** | Strategy discussion, not the notebook | `src/zoneboost/_categorical_depth.py` (`CategoricalDepthTransformer`) |
| **ZoneForest** (bagged ensemble of shallow zone models) | Shipped | Notebook page 3 ("breaks down data to smaller samples & averages it out") | `src/zoneboost/_zone_forest.py` (`ZoneForest`) |
| **Cythonize zone lookup/split search** | Proposed | Strategy discussion, not the notebook | Performance work inside `_zones.py`/`_weak_learner.py`'s existing lookup/split-search paths; no API change, no new opacity |
| **ZoneFeatureSpace** (first-class transformer/feature-space API) | Shipped | Notebook page 2 ("1. the ado-transformer") | `src/zoneboost/_feature_space.py` (`ZoneFeatureSpace`) |
| **Correlation-aware zone boundaries** | Shipped | Notebook page 2 ("correlation", "covariance") | `src/zoneboost/_zones.py` (`split_criterion="correlation"` on `adaptive_zone_boundaries`), threaded through `ZoneProfileEncoder`/`ConditionalZoneGrid`/`ZoneFeatureSpace` |
| **zoneboost.eda** (`zone_boxplot`, `drift_dashboard`) | Shipped | Notebook page 2 ("compare outcome & variables boxplots") | `src/zoneboost/eda/` (`zone_boxplot`, `drift_dashboard`), gated behind the `zoneboost[eda]` extra |
| **ZoneBoostTimeSeries** (native expanding/rolling walk-forward fitting) | Shipped | Notebook page 3 ("sequential date pattern") | `src/zoneboost/_time_series.py` (`ZoneBoostTimeSeries`); fits one model per period and reuses `compare_models`/`flag_drift` across consecutive periods automatically |
| **Signed contribution waterfall** (`prediction_waterfall`, `signed_contribution_profile`) | Shipped | Notebook page 2 ("min-max scaler — direction included") | `src/zoneboost/eda/_prediction_waterfall.py`, `src/zoneboost/eda/_signed_contribution_profile.py` |
| **Spline zones** (bounded linear trend within a zone) | Proposed | Notebook page 1 ("genuinely continuous relationship") | Would extend `_weak_learner.py`'s per-zone constant mean to an optional per-zone linear fit, still bounded by zone edges — not a recursive split |

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

**Laplace history transformer.** `LaplaceHistoryTransformer(entity_col=,
time_col=, half_lives=, value_columns=None, group_name=None)` gives a
model memory of a per-entity event history (claims, purchases, fraud
events, sensor readings) via exponential half-life decay, without a
recurrent/attention architecture. Structurally unlike every other
transformer above: those turn static columns of `X` into new columns of
the same `X` with a plain `transform(X)` that composes automatically
inside a `ColumnTransformer`/`Pipeline`; this one needs a second,
long-format event log at `fit` (`fit(X, y=None, event_history=...)`, kept
out of `__init__` to preserve sklearn's own `clone()`/`get_params()`
convention) and a per-row asof cutoff at `transform`
(`transform(X, asof_col=..., cutoff_date=...)`, exactly one required), so
it's used standalone rather than dropped into an automatic pipeline step.
The original proposal's API took one global `cutoff_date` applied to every
row -- refined during scoping to a **per-row** `asof_col` instead, since a
single shared cutoff is leakage-prone for a historical training set built
from rows at many different past dates (it would either leak future
events into early rows or go stale on recent ones); `cutoff_date` survives
as a convenience that broadcasts one asof to every row for the
single-shared-moment case (e.g. production batch scoring), still filtered
through the identical per-row lookup. For each half-life, emits a
decay-weighted sum per `value_columns` entry plus a decay-weighted event
count (the amount and frequency halves of a standard RFM-style feature,
sharing one computation -- added during scoping, not in the original
proposal) and a `"..._had_history"` cold-start flag (`0` for an entity
never seen in `event_history`, whose decay columns are then `0.0`, not
`NaN`). Because the leakage guard (`time_col <= asof`) is enforced
independently per output row, one fitted `event_history` -- which may
contain events later than some rows' own asof -- is safe to reuse across
an entire historical training set without slicing it per fold first. See
`src/zoneboost/_laplace_history.py`, `README.md` ("Laplace history
transformer"), `docs/how-it-works.html` (`#laplace-history-transformer`),
`docs/api-reference.html` (`#laplace-history-parameters`).

**Depth crowd.** `DepthCrowd(columns=None, rank_normalize=True,
vote_threshold=0.05, group_name="crowd")` aggregates several already-
computed per-expert typicality scores (e.g. several `DepthTransformer`
instances' `*__coreness` columns from a `FeatureUnion`, one per domain
zone) into a crowd-level mean/median/std/min, a vote count/share, and
which expert is driving an outlier for each row. Scoped down sharply from
a much larger "wisdom of the crowd" pitch (bootstrap-covariance crowding,
random-subspace zone generation, class-conditional centers, per-expert
learned reliability weights, and a full stacked meta-model over
probability outputs) after checking it against what's already free or
already exists: multiple domain-zone experts are already achievable today
via `DepthTransformer` + `FeatureUnion` with zero new code; supervised
soft-voting/stacking is exactly scikit-learn's own
`VotingClassifier`/`StackingClassifier`; bootstrap-covariance crowding
duplicates `BootstrapStability`'s resampling machinery and was left as a
distinct, separate future feature rather than folded in here. What
survived scoping was the one real technical gap the original pitch
flagged but never resolved: `DepthTransformer.coreness` is explicitly
documented as not a calibrated percentile, so raw `coreness` values from
differently-scoped experts (different column counts/covariance structure)
aren't directly comparable. `rank_normalize` (default on) fixes this
properly with a **fitted** reference distribution per expert column
(stored at `fit`, looked up via `np.searchsorted` at `transform` -- the
same technique `ConditionalZoneGrid`/`LaplaceHistoryTransformer` already
use), rather than normalizing against whatever rows happen to be in the
current `transform` call. Unlike `LaplaceHistoryTransformer`, this one
fits the plain `fit(X, y=None)`/`transform(X)` shape every other
transformer here uses (its input is a normal per-row feature table, not a
second event-log table), so it composes automatically inside a `Pipeline`
right after the `FeatureUnion` that produces its input.
`crowd__most_atypical_expert` is a new pattern for this codebase -- a
string label column (every prior transformer emits numeric-only output),
disclosed as meant for human review/audit, droppable before feeding the
rest into a model that needs purely numeric input; kept always-on rather
than an opt-out flag, the same precedent `ConditionalZoneGrid`'s
`used_segment_grid` disclosure column already sets. See
`src/zoneboost/_depth_crowd.py`, `README.md` ("Depth crowd"),
`docs/how-it-works.html` (`#depth-crowd`), `docs/api-reference.html`
(`#depth-crowd-parameters`).

**Categorical depth transformer.** `CategoricalDepthTransformer(columns=None,
group_name=None)` is the discrete sibling of `DepthTransformer`, prompted
by a direct question -- does `DepthCrowd`/`DepthTransformer` work well
with binary or categorical data -- answered by testing rather than
assuming: `DepthTransformer` explicitly rejects a declared categorical/
`bool` column, and silently accepts an int-coded 0/1 column but produces a
coarse, barely-discriminating `coreness` for it (confirmed: mixing one
binary column collapsed 200 rows down to 185 distinct coreness values),
because Mahalanobis distance's geometry assumes roughly continuous,
elliptical structure a discrete variable doesn't have. Reuses existing,
already-tested primitives rather than inventing new categorical-bucketing
logic: `categorical_zone_map`/`categorical_zone_index`
(`src/zoneboost/_zones.py`) already give each distinct value its own zone
with two separately-reserved fallback zones (missing vs. unseen-but-real)
for free; multiple declared columns combine into one joint cell index via
mixed-radix encoding, the same `combined = za * n_b + zb` trick
`ConditionalZoneGrid._fit_grid` already uses for 2 columns, generalized
here to however many columns are declared. Emits the joint cell's raw
training support count (`"<group>__count"`) and that count as a fraction
of the training set (`"<group>__coreness"`, bounded `[0, 1]`, higher =
more typical) -- deliberately matching `DepthTransformer`'s own
raw-then-bounded disclosure pattern and column-suffix convention, so a
caller building a `DepthCrowd` `columns=[...]` list doesn't need to
remember two different naming schemes for a continuous expert vs. a
categorical one. No shrinkage toward a prior, unlike `ConditionalZoneGrid`'s
cell mean of y: a raw count is already an honest answer even at `count =
1`, there's nothing to shrink toward. No automatic fallback for
combinatorial sparsity either, a real, disclosed difference from
`ConditionalZoneGrid`'s `min_segment_size` fallback -- that fallback
exists because a *mean* needs enough support to trust, which doesn't apply
to a raw count. See `src/zoneboost/_categorical_depth.py`, `README.md`
("Categorical depth transformer"), `docs/how-it-works.html`
(`#categorical-depth-transformer`), `docs/api-reference.html`
(`#categorical-depth-parameters`).

**ZoneForest.** `ZoneForest(base_estimator=None, n_estimators=100,
max_samples=1.0, max_features=1.0, n_jobs=None, random_state=42)` is the
"averaging" paradigm the notebook describes, distinct from
`ZoneBoostRegressor`'s sequential-boosting-on-one-sample paradigm: bags
`n_estimators` clones of `base_estimator` (default a plain
`ZoneBoostRegressor()`), each fit on its own bootstrap resample (rows with
replacement -- same convention as `BootstrapStability`) and its own fixed
column subset (drawn once per estimator, without replacement), dispatched
via `joblib.Parallel`. `predict`/`explain`/`feature_importance` all reduce
to plain means across the ensemble -- `explain(X)` still sums exactly to
`predict(X)` (a term some estimators never fit, because column
subsampling excluded a constituent column, contributes exactly `0` for
those estimators, not a re-normalized share -- linearity of the mean
preserves the exact-sum guarantee). `base_estimator.group_col`/
`mondrian_col` are force-included in every estimator's column subset
regardless of `max_features`, the same "never subsampled away" precedent
`ZoneBoostRegressor.group_col` already sets against its own internal
`col_subsample` -- those two params are the only ones that would otherwise
raise `ValueError` from a base estimator whose own view of `X` is simply
missing a declared column; every other column-selecting param (
`categorical_features`, `monotonic_constraints`, ...) already degrades
silently when a declared column is absent, so isn't specially handled.
Measured honestly rather than assumed: bagging shallow learners here
turned out to be a genuine variance-reduction and parallel-speed win, not
an accuracy win over a single well-tuned deep `ZoneBoostRegressor` fit --
worse still, `max_features < 1.0` measurably hurt RMSE on data with a
genuine interaction (a bagged member missing either of an interaction's
two constituent columns can't see that interaction at all), a real,
disclosed bias/diversity tradeoff rather than a free regularization.
Scoped down from the original pitch to regressor-only (a
`ZoneBoostClassifier` `base_estimator` -- majority vote or averaged
`predict_proba` -- is a separate, larger aggregation change, deferred)
and with no native prediction-interval/spread method of its own
(uncertainty stays `BootstrapStability`'s job; compose the two,
`BootstrapStability(ZoneForest(...))`, if both are wanted, rather than a
second overlapping API for the same kind of question). See
`src/zoneboost/_zone_forest.py`, `README.md` ("ZoneForest"),
`docs/how-it-works.html` (`#zoneforest`), `docs/api-reference.html`
(`#zoneforest-parameters`).

**ZoneFeatureSpace.** `ZoneFeatureSpace(zone_profiles=True,
categorical_features=None, depth_scores=False,
categorical_depth_scores=False, conditional_grids=None, ...)` is a thin,
convenience-first orchestrator over transformers that were already
`sklearn` `TransformerMixin`s composable via a plain `FeatureUnion` with
zero new code (`ZoneProfileEncoder`, `DepthTransformer`,
`CategoricalDepthTransformer`, `ConditionalZoneGrid`) -- checked against
"what's already free" before building new API surface, the same
discipline `DepthCrowd`'s own scoping applied. What survived that check as
genuinely new: `explain(values)`, which rolls a downstream model's
`coef_`/`feature_importances_` (aligned to `get_feature_names_out()`)
back to each **original raw column**, tracked via each sub-transformer's
own `columns_`/`segment_columns_` attributes at `fit` time rather than
parsed from the output column name string (a raw column like
`"annual_income"` would make name-parsing genuinely ambiguous against the
`__` suffix convention every transformer here shares) -- an output column
with more than one source contributes its full, undivided magnitude to
every one of its source columns, the same "never divided back between
parent variables" rule `ZoneBoostRegressor.explain()` itself follows. Also
new: `suggest_interactions(X, y, columns=None, top_k=10)`, a deliberately
scoped-down answer to the original pitch's `interaction_candidates="auto"`
-- checked whether `_weak_learner.py`'s own cross-fitted pair-screening
proxy (`_pair_interaction_score`) could be reused directly, and it
couldn't cleanly: it's entangled with a boosting round's own zone
construction, cross-fitting folds, and residual, so exposing it standalone
would mean either duplicating real internal machinery or building a
cruder heuristic and disclosing that it's cruder. Chose the latter,
honestly: scores each candidate pair by the absolute correlation between
an OLS `y ~ a + b` fit's residual and the centered product `(a - mean(a))
* (b - mean(b))`, returns ranked tuples **to review**, never auto-fits a
grid, never guesses `segment_columns` (a separate, harder problem).
`depth_scores`/`categorical_depth_scores` accept `True` (one instance,
matching that transformer's own single-joint-group default) or a list
where a plain `list` entry is an auto-named group and a `(name, columns)`
tuple is an explicitly named one -- list-vs-tuple as the disambiguator,
chosen over type-sniffing column contents. Deferred, disclosed: `DepthCrowd`
(aggregates *already-built* depth outputs -- an aggregator-of-aggregators,
not a `columns -> features` step this class's flat toggles can represent)
and `LaplaceHistoryTransformer` (needs a second `event_history` table and
a per-row `asof_col` at transform -- a fundamentally different signature).
See `src/zoneboost/_feature_space.py`, `README.md` ("ZoneFeatureSpace"),
`docs/how-it-works.html` (`#zonefeaturespace`), `docs/api-reference.html`
(`#feature-space-parameters`).

**Correlation-aware zone boundaries.** `adaptive_zone_boundaries`
(`_zones.py`) gains an opt-in `split_criterion="variance"|"correlation"`
parameter. `"variance"` (the default, reproducing every prior release
bit-for-bit -- verified, not just argued) is the existing regression-tree-
style criterion: each cut most reduces `y`'s within-segment sum of
squares. `"correlation"` instead prefers a cut where the local OLS slope
of `y` on the column **reverses sign** left vs. right of it -- a genuine
regime change (a threshold effect, a U-shape's vertex), not just a level
shift, computed via the same vectorized running-sums technique
`_best_split` already uses (three more cumulative sums: `Sx`, `Sxx`,
`Sxy`), so it stays the same complexity class, not asymptotically worse.

A design correction made *before* writing any code, not after: the
naive version (per-segment, try a correlation split then fall back to a
variance split) would have compared two genuinely incommensurate gain
scales directly (slope units vs. sum-of-squares units) in the outer
recursive loop's cross-segment "pick the single best gain" comparison --
caught during scoping and fixed with a two-phase, lexicographic rule
instead: each iteration first scans *every* open segment for a genuine
sign-reversal candidate and, if any exists, splits the best one by its
own correlation-mode gain; only once **no** open segment has a reversal
candidate does the loop fall back to ordinary variance-reduction splits
for whatever remains -- the two gain scales are never compared against
each other, only ever against their own kind.

Scoped down deliberately on two axes, both decided by asking rather than
assuming: threaded through `ZoneProfileEncoder`/`ConditionalZoneGrid`/
`ZoneFeatureSpace` only, **not** `ZoneBoostRegressor`'s own per-round
split search (`_weak_learner.py`, called every round for every column --
by far the highest-frequency, highest-blast-radius call site) -- the same
precedent `ConditionalZoneGrid` itself set by shipping standalone before
any boosting-loop integration; and a segment with no genuine reversal
falls back to a plain variance split rather than leaving zero splits on a
monotonic column, a real (disclosed) blend rather than a strict
"every boundary here is a proven reversal" guarantee. No significance
test on the slope (no t-statistic/p-value) -- a cheap heuristic (the
slope must come out exactly nonzero after a guarded division), disclosed
as such, not a calibrated test for whether a reversal is real versus
sampling noise.

Measured, not just argued, before documenting it: on a sharp V-shape
(`y = |x - 2| + noise`, a genuine abrupt regime switch), `"correlation"`
placed its boundary at `2.13` (error `0.13` from the true kink at `x=2`)
versus `-4.41` (error `6.41`) for `"variance"`. On a smooth parabola
(`y = (x - 2)^2 + noise`), where the local slope itself vanishes exactly
at the true vertex, `"correlation"` landed at `4.07` (error `2.07`)
versus `-5.76` (error `7.76`) for `"variance"` -- still clearly better,
but visibly less precise than the sharp-kink case, an honest, disclosed
characteristic (best for abrupt/threshold-like reversals) rather than a
universal "finds the exact reversal point" claim. See
`src/zoneboost/_zones.py`, `README.md` ("Correlation-aware zone
boundaries"), `docs/how-it-works.html`
(`#correlation-aware-zone-boundaries`), and the `split_criterion` row in
the `ZoneProfileEncoder`/`ConditionalZoneGrid`/`ZoneFeatureSpace`
parameter tables in both `README.md` and `docs/api-reference.html`.

**zoneboost.eda.** `zone_boxplot(X, y, column, n_zones=7, min_zone_frac=
0.02, min_zone_abs=20, split_criterion="variance", valid_range=None,
plot=True, ax=None)` and `drift_dashboard(model_old, model_new, X_eval,
y_eval=None, plot=True, top_n=15)` are a new `zoneboost.eda` subpackage --
a **separate** subpackage, not part of the top-level `zoneboost`
namespace, because this is the first feature needing a dependency beyond
`numpy`/`pandas`/`scikit-learn` for something other than an LLM call:
`matplotlib`, gated behind a new `zoneboost[eda]` extra, imported lazily
inside each function's own body (never at module import time -- verified
directly: `import zoneboost.eda`/`from zoneboost.eda import zone_boxplot`
import only `numpy`/`pandas`/zoneboost's own `_zones.py`/`_drift.py`, the
same "only import inside the method body that actually needs it"
discipline `LLMZoneNamer` already established for `anthropic`).

Neither function invents new modeling math -- deliberately, the same
compose-rather-than-build-in precedent every transformer in this ledger
already sets. `zone_boxplot` reuses the *exact same* zone construction
`ZoneProfileEncoder` already calls (`adaptive_zone_boundaries`/
`zone_index` for continuous columns, `categorical_zone_map`/
`categorical_zone_index` for categorical ones, `split_criterion` forwarded
straight through so it composes with "Correlation-aware zone boundaries"
above for free) rather than a separately-invented quantile-bin boxplot.
`drift_dashboard` calls `compare_models` internally and computes zero new
statistics of its own -- `plot=False` returns *exactly* `compare_models`'s
own dict, a pure passthrough, not a re-derived summary.

`valid_range=(lower, upper)` on `zone_boxplot` answers the notebook's own
"business would know better" framing directly, scoped down from a vaguer
"mark business-valid vs invalid" pitch to the same "declare it, don't
infer it" convention `monotonic_constraints`/`bounded_effects` already use
on `ZoneBoostRegressor` -- a numeric bound is domain knowledge no
statistic can infer on its own, so it's caller-supplied, not guessed.
Emits a `pct_business_invalid` column only when actually declared (a zone
straddling the boundary gets a real fraction between 0 and 1, not a
binary flag), and raises `ValueError` if declared on a categorical column
(a numeric bound has no meaning there, the same rejection
`DepthTransformer` already gives a declared categorical column for an
analogous reason). The always-on statistical signal --
`outlier_count`/`outlier_frac` per zone, the standard IQR boxplot rule
applied to `y` within that zone -- needs no such declaration and is
present regardless.

`drift_dashboard`'s multi-panel figure needs one number `compare_models`
itself doesn't expose beyond a `{"mean", "std"}` summary -- the raw
per-row `predict_new(X) - predict_old(X)` array for its histogram panel
-- so that one panel recomputes it directly via two `predict` calls,
disclosed in both the docstring and the rendered figure's own construction
rather than silently reaching past `compare_models`'s documented return
shape. Gracefully omits the boundary-shift/population-migration panels
entirely (rather than rendering an empty one) when the two models share no
continuous main-effect column, e.g. two purely-categorical models.

Every function still returns its full underlying numbers (`stats`
DataFrame, or `compare_models`'s own dict) even when `plot=True` also
draws a picture -- nothing here is only inspectable as a rendered image,
the same disclosure-over-convenience precedent this whole ledger has kept
throughout. **Scope**: static `matplotlib` figures only (no interactive/
HTML charting backend, e.g. plotly); no automatic light/dark theming (a
single rendered image, not a themed report -- deliberately out of scope
for a plotting utility, unlike an HTML artifact). See
`src/zoneboost/eda/__init__.py`, `src/zoneboost/eda/_zone_boxplot.py`,
`src/zoneboost/eda/_drift_dashboard.py`, `README.md` ("zoneboost.eda
(optional)"), `docs/how-it-works.html` (`#zoneboost-eda`),
`docs/api-reference.html` (`#zone-boxplot-signature`,
`#drift-dashboard-signature`).

**ZoneBoostTimeSeries.** `ZoneBoostTimeSeries(base_estimator=None,
time_col=None, freq="Y", window="expanding", min_periods=2,
random_state=42)` fits one `ZoneBoostRegressor` per time period,
walk-forward, and automatically compares every consecutive pair via
`compare_models`/`flag_drift` -- both already shipped, so this module
invents zero new modeling math, purely period-bucketing/windowing
orchestration around estimators/functions that already exist. Two real
scope decisions were resolved by asking rather than assuming (the
original pitch's own phrasing left both ambiguous):

The original pitch's three window strings (`"expanding"`/`"rolling"`/
`"walk_forward"`) collapsed to two (`"expanding"`/`"rolling"`) -- walk-
forward evaluation (comparing a period's own model against the next
period's genuinely held-out data) turned out to be inherent to *either*
window definition once fit per-period, not a third, distinct way to
define the training window itself; keeping a separate `"walk_forward"`
string would have meant two parameter values behaving identically, worse
than not offering a third option. `min_periods` reuses one integer-count
knob for both window modes rather than adding a second, overlapping
`window_periods` parameter: for `"expanding"` it's the minimum periods
pooled for the first fit (the window then only grows); for `"rolling"`
it's the fixed size every window keeps. Deliberately an integer period
**count**, not a timedelta string like the original pitch's own `"2Y"`
phrasing -- `freq` (a plain pandas offset alias, resolved via pandas'
own `.dt.to_period(freq)`, no date-parsing logic reinvented here)
already controls what one period spans, so `min_periods=2` with
`freq="Y"` already means "2 years," with no second string-parsing
convention needed on top.

`time_col` is excluded from every per-period model's own fitted feature
space entirely -- not optional, always: a raw timestamp has no meaning
as a zone-averaged predictor, and `resolve_categorical_features` would
otherwise auto-detect a datetime64 column as an absurdly high-cardinality
"categorical" column (confirmed by reasoning through
`_common.py`'s own `is_numeric_dtype` check before writing any code, not
discovered by trial and error afterward). A caller who wants the
calendar signal itself fitted (day-of-year, a trend index) engineers
that as a separate column first -- this class derives no calendar
features of its own.

`predict(X)`/`explain(X)`/`feature_importance(X)` all delegate to the
most recent period's own fitted model (`models_[max(models_)]`) -- a
genuine, usable `RegressorMixin`, not a diagnostic-only wrapper like
`BootstrapStability`, decided explicitly since a caller should be able to
deploy the fitted object directly for production scoring, with every
per-period diagnostic available as additional attributes on top rather
than the only thing this class does. `drift_alerts_` only contains a
consecutive pair when the newer period's own model has a non-`None`
`conformal_scores_` (i.e. was fit with `validation_fraction > 0` or
`calibration_fraction > 0`, the `ZoneBoostRegressor` default) -- a pair
missing from `drift_alerts_` while still present in `comparisons_`
discloses that precondition wasn't met, the same "membership itself is
the disclosure" precedent `ConditionalZoneGrid.segment_grids_` already
sets for a segment below `min_segment_size`.

`stability_report()` answers the original pitch's own "detect which
zones are stable vs. drifting over time" ask directly: an aggregation of
`compare_models`'s own per-pair `population_migration`/`boundary_shift`
numbers across every consecutive comparison, per feature -- zero new
statistics, sorted by mean population migration descending. Verified,
not just argued: on synthetic data with one feature whose relationship
to `y` genuinely flips sign halfway through the series and one whose
relationship never changes, the flipping feature ranked first.

Two related asks from the original pitch were deliberately **not**
attempted here, disclosed rather than half-built: "flag when a zone's
empirical Bayes prior should be updated" has no existing defensible,
general rule to build on (`drift_alerts_`/`stability_report()` are the
closest available signal, not a replacement for a real rule that doesn't
exist yet); "generate the boxplots over time" is left to the caller via
`zoneboost.eda.drift_dashboard(model.models_[k], model.models_[k_plus_1],
X_eval, y_eval)` per consecutive pair directly, rather than adding a
second, ZoneBoostTimeSeries-specific plotting entry point on top of
`zoneboost.eda`'s own already-shipped one. **Scope**: regressor only
(`compare_models` itself is regressor-only). See
`src/zoneboost/_time_series.py`, `README.md` ("ZoneBoostTimeSeries"),
`docs/how-it-works.html` (`#zoneboosttimeseries`),
`docs/api-reference.html` (`#time-series-parameters`).

**Signed contribution waterfall.** The original pitch's own single
function name (`plot_signed_contributions(X, feature="income")`) turned
out to describe two genuinely different charts once its own illustrative
text was read closely -- its example ("high income adds +$120, but if
age > 65 in the Northeast, subtract $45") mixes two *different* terms in
one scenario, which reads as a per-prediction decomposition, while the
signature itself (`feature="income"`) reads as a per-feature profile.
Rather than guess which one, both were asked about and both were
approved, shipping as two functions in `zoneboost.eda`:
`prediction_waterfall(model, X, index=0, max_terms=None, purify=False,
...)` -- every term's own contribution to *one row's* prediction, sorted
by `|contribution|` descending, stacked from `baseline` to the final
predicted value -- and `signed_contribution_profile(model, X, feature,
n_zones=7, ..., split_criterion="variance", purify=False, ...)` -- how
*one feature's* own main-effect contribution rises or falls across its
own range, zone by zone. Neither invents new modeling math:
`prediction_waterfall` is a pure ordering/rendering layer over
`explain(X.iloc[[index]])`'s own already-exact row; `signed_contribution_profile`
reuses the *exact same* zone construction `zone_boxplot` already uses
(`adaptive_zone_boundaries`/`categorical_zone_map`, `split_criterion`
forwarded through for free), just binning against `feature`'s own
already-computed `explain(X)` contribution column instead of the real
target `y` -- the identical machinery, a different "y."

`max_terms` on `prediction_waterfall` folds the smallest-magnitude excess
terms into one `"other"` row (summed, not dropped) once a model has more
terms than fit readably on one chart -- verified, not just argued: the
waterfall's own `"total"` row matches `predict()` to within floating-point
noise whether or not `max_terms` truncates. A real bug was caught and
fixed during this same implementation pass, before ever committing:
both functions initially forwarded `purify=purify` to `model.explain(...)`
unconditionally, but `ZoneBoostClassifier.explain()` has no `purify`
parameter at all (purification is regressor-only) -- calling either
function on *any* classifier, not just a multiclass one, would have
raised a confusing `TypeError` regardless of what the caller actually
passed for `purify`. Fixed by checking `inspect.signature(model.explain)`
for a `purify` parameter before forwarding it, raising a clear,
intentional `ValueError` only if the caller explicitly asked for
`purify=True` on a model that can't honor it. `signed_contribution_profile`
requires `feature` to have appeared as its own main-effect term in
`explain(X)` -- raises `ValueError` otherwise (a column only ever fit
inside an interaction, or never sampled by `col_subsample` in any round,
has no main-effect contribution column of its own to profile; verified
directly with a constructed single-round, forced-column-exclusion fit
rather than assumed). Both raise `ValueError` for a multiclass classifier
(`explain(X)` returns a `{class_label: DataFrame}` dict there, not the
single flat table either function needs). See
`src/zoneboost/eda/_prediction_waterfall.py`,
`src/zoneboost/eda/_signed_contribution_profile.py`, `README.md`
("zoneboost.eda (optional)"), `docs/how-it-works.html`
(`#zoneboost-eda`), `docs/api-reference.html`
(`#prediction-waterfall-signature`,
`#signed-contribution-profile-signature`).
