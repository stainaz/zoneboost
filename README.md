# zoneboost

Fully transparent, zone-based gradient boosting — no decision trees, no
gradient descent, no neural weights. Every number in a prediction traces
back to a quantile, a group count, or a group average, and is inspectable
directly from the fitted model.

Two estimators, sharing the exact same weak learner: `ZoneBoostRegressor`
and `ZoneBoostClassifier` (binary and multiclass). Both are
scikit-learn-compatible: they work with `Pipeline`, `GridSearchCV`,
`cross_val_score`, and `clone`.

![Overview of zoneboost's mechanism: zones, per-zone scoring, pairwise interactions, boosting rounds, and a worked prediction with its exact contribution breakdown](docs/assets/images/zoneboost-explanation.png)

## Installation

```bash
pip install zoneboost
```

## Quickstart

```python
import pandas as pd
from zoneboost import ZoneBoostRegressor

X = pd.DataFrame({
    "rooms": [3, 4, 2, 5, 3, 4, 2, 5],
    "distance_km": [5.0, 2.0, 8.0, 1.0, 6.0, 3.0, 7.5, 1.5],
    "neighborhood": ["a", "b", "a", "b", "a", "b", "a", "b"],
})
y = [300, 450, 220, 520, 310, 470, 230, 510]

model = ZoneBoostRegressor(categorical_features=["neighborhood"], random_state=0)
model.fit(X, y)
model.predict(X)
```

```python
from zoneboost import ZoneBoostClassifier

y_class = [0, 1, 0, 1, 0, 1, 0, 1]
clf = ZoneBoostClassifier(categorical_features=["neighborhood"], random_state=0)
clf.fit(X, y_class)
clf.predict_proba(X)   # (n_samples, n_classes), rows sum to 1
clf.predict(X)          # works for binary and 3+ classes (one-vs-rest) alike
```

## How it works

Each boosting round fits a "weak learner" made of two transparent pieces,
both built by splitting each predictor's axis into a small number of
data-driven zones and averaging the current residual within each zone (or
zone pair):

- **Main effects** — for each predictor, a 1D lookup from zone to average
  residual.
- **Interactions** — for every pair of predictors, a 2D lookup from their
  joint zones to average residual, capturing effects neither variable
  explains alone.

**Continuous** predictors get *adaptive* zone boundaries, found the way a
regression tree finds a split — the cut that most reduces the target's
within-zone variance — re-derived fresh every round from that round's
residual, rather than fixed quantile bins computed once.

**Categorical** predictors (declared via `categorical_features`, or
auto-detected from `object`/`category`/`bool` dtype) skip that search
entirely: every distinct value gets its own zone. A cut-point search
assumes two values that are numerically close behave alike — true for a
continuous variable, false for a nominal category like a neighborhood ID,
where there's no reason two adjacent label-encoded values behave similarly.

Every zone's own mean is shrunk toward a hierarchical prior via
**empirical Bayes** (see "Empirical Bayes shrinkage" below) — so sparse
zones lean toward their prior instead of overfitting a handful of rows.
Each round's correction is applied at a small, shrunk step
(`learning_rate`) and added to a running prediction, exactly like standard
gradient boosting. `row_subsample` / `col_subsample` add
stochastic-gradient-boosting-style regularization by fitting each round on
a random subsample of rows and columns.

### Missing values

Both continuous and categorical columns accept NaN/None directly — no
imputation needed beforehand. A missing value gets its own dedicated zone,
kept separate from an unseen-but-real category (a value that exists but
wasn't present at fit time), the same way an unseen category is handled.
If missingness itself is informative (a common, real phenomenon — e.g. a
sensor reading that's absent exactly when it would have been extreme), the
model learns that: the missing zone gets its own data-driven average
contribution from whichever training rows were actually missing for that
column, rather than being silently imputed away or corrupting the
adaptive split search for the column's present values.

### Classification

`ZoneBoostClassifier` uses the *identical* weak learner — same main
effects, same interactions, same empirical-Bayes shrinkage. The only
change is where boosting happens: each round is fit against the residual in
**log-odds space** (`y - sigmoid(current_score)`, the standard logistic-loss
gradient) instead of the raw target, and predictions are squashed through
a sigmoid at the end. This is the standard way gradient boosting
generalizes from regression to classification.

Binary targets fit a single log-odds booster. 3+ classes use one-vs-rest:
an independent booster is fit per class ("is this class vs. everything
else"), sharing one validation split across all of them, and their
probabilities are normalized to sum to 1 at predict time — multiclass is
not a different mechanism, just K independent copies of the same binary
booster.

### Adaptive interaction order

By default zoneboost learns main effects and every pairwise interaction
(`max_interaction_order=2`). Setting `max_interaction_order=3` additionally
attempts a bounded, adaptive search for 3-way interactions each round:
candidates are seeded from the columns appearing in that round's strongest
pairs (not every possible `(a, b, c)` triple, which would be
combinatorially expensive), and a candidate is only kept if a joint 3-way
zone grouping still explains meaningful residual variance beyond what main
effects and its three constituent pairwise interactions already predict —
evidence of a genuine higher-order pattern, not something pairwise terms
already cover. Surviving candidates are ranked by that evidence and only
the strongest `max_triple_interactions` are kept per round. If nothing
clears the bar in a given round, no triples are added that round — this is
why the default (`max_interaction_order=2`) produces identical models to
every prior release: the 3-way search is strictly opt-in.

### Cross-fitted cell means

Every zone's mean (main effect, pairwise, or 3-way) is otherwise computed
from the same rows a round then scores — each row's own residual partly
determines the zone mean it's then judged against, the same in-sample
leakage CatBoost's ordered boosting was built to fix. Left alone, this
biases the boosting trajectory optimistic about sparse zones (small
`min_zone_frac` continuous zones, high-cardinality categoricals), since a
zone with a handful of rows can end up mostly reconstructing its own
members' values rather than reflecting real structure.

Each round instead splits its (already row/column-subsampled) rows into
`cross_fit_folds` folds (default 5) and scores each fold only with zone
tables built from the *other* folds — no row is ever scored with a table
that included its own value. Only the training signal is affected; the
tables actually stored in `rounds_` and used by `predict`/`explain` still
use every available row, since new data was never part of the leakage to
begin with. This is on by default (not a `max_interaction_order`-style
opt-in) — it's a correctness fix, not a feature.

Cross-fitting also exposed a related fragility worth knowing about: a
round's raw zone-lookup score can no longer be rescaled to the residual's
units via a std-ratio (`resid_mean + (raw - raw_mean) * (resid_std /
raw_std)`), since that forces `raw`'s spread to match the residual's
regardless of how well the two actually correlate — harmless when `raw`
is in-sample-inflated (as it always was pre-cross-fitting), but once
cross-fitting honestly reveals a round found no real signal, `raw`'s
variance can legitimately collapse toward zero, and dividing by a
near-zero value amplifies noise instead of correctly damping it. This was
first fixed with an ordinary-least-squares rescale, later superseded by
the Lasso fit described next (which has the same non-amplifying property,
plus per-term weights instead of one shared scale).

### Empirical Bayes shrinkage

Every prior release weighted a zone's contribution by
`confidence = counts / counts.max()` — a flat, ad hoc discount relative to
that round's busiest zone. This is replaced by an **empirical-Bayes
(m-estimate) shrinkage** of the zone's mean itself:

```
shrunk_mean = (n · cell_mean + m · prior) / (n + m)
```

A zone needs about `m` rows of its own (`shrinkage_m`, default 10) before
it's trusted as much as its prior; fewer rows lean toward the prior, more
rows lean toward its own data. Critically, the prior is **hierarchical**,
not the flat global mean:

- **Main effects** shrink toward the global mean.
- **Pairwise interactions** shrink toward the additive combination of
  their own row and column marginals (each already shrunk the same way) —
  for a sparse joint cell, "what row A's zone alone predicts, plus what
  column B's zone alone predicts" is a far better guess than the overall
  average of everything.
- **3-way interactions** shrink one level deeper still, toward the
  additive combination of their three main effects and three pairwise
  interactions (all already shrunk).

This fully replaces the confidence mechanism rather than supplementing
it — once a cell's own mean is properly shrunk in proportion to how little
data supports it, a separate trust-discount multiplied on top is
redundant. Like the cross-fitting fix, this is on by default: a more
principled estimate, not a `max_interaction_order`-style opt-in.

### Lasso stacking

Every prior release combined a round's terms by averaging every
contribution equally (`raw = contributions.mean(axis=1)`), then fit one
shared scale for the whole blend — every term got the same diluted
`1/n_terms` weight regardless of relevance. This is replaced by a
**Lasso** fit that treats each term's own (cross-fitted) contribution as
its own feature:

- An irrelevant term's weight gets zeroed by the L1 penalty.
- A strong term gets its own learned weight instead of a diluted share.
- The fitted weights are themselves a real interaction-importance
  ranking, flowing straight through `feature_importance()`/`explain()`
  with no new API needed.

Both sides are standardized before fitting (each contribution by its own
std, the residual by its own) so `stacking_alpha` — the L1 regularization
strength — is unitless and comparable across rounds and datasets. On a
dataset with one genuine interaction mixed among several irrelevant
columns, equal-weight averaging diluted the real signal into an
unrecoverable blend (test R² ≈ 0); Lasso stacking recovered it cleanly
(test R² > 0.85) — the gap the reviewer's roadmap predicted this would
close. Like the two changes above, this is on by default.

### Soft zone boundaries

Continuous zone boundaries were hard cuts: a value one unit below a cut point and one
unit above it land in completely different zones with independently-shrunk means — a
"cliff edge" discontinuity in the prediction at the exact boundary, which doesn't match
how a genuinely continuous relationship should behave. Every real zone now also gets a
**centroid** — the empirical mean training-x-value of the rows that landed in it — and a
lookup blends between a value's own zone and whichever neighboring zone its centroid
points toward, rather than hard-assigning it to exactly one:

- 0 exactly at its own zone's centroid, 1 exactly at the neighbor's, linear between,
  clamped past either end (leftmost/rightmost zone, or a single-zone column) so it never
  reaches past a non-existent neighbor.
- Main effects become a 2-point linear blend; pairwise interactions become the standard
  4-corner bilinear blend; triples become the 8-corner trilinear analog.
- Categorical columns and missing values are an exact no-op (always fully their own hard
  zone) — there's no meaningful "distance" to interpolate along a nominal category, so
  only continuous-column lookups change. A pair/triple with a categorical member
  naturally interpolates only along its continuous member(s).

Zone *construction* (the adaptive split search, `min_zone_frac`, `max_zones`) and each
zone's own fitted mean (still computed by hard-grouping training rows, unchanged) are
untouched — only how a value is *looked up* against an already-fitted grid changes. On a
sharp step function, the largest single-step prediction change across an infinitesimal
step over the true boundary dropped from ~3.9 (almost the full step size) to ~0.2 — and,
consistent with "helps generalisation," a continuous-interaction test case's held-out R²
improved further on top of what cross-fitting/shrinkage/stacking already delivered. Like
the three changes above, this is on by default; there's no natural partial-strength knob
to expose as a parameter, so no new one was added. Fitting cost is meaningfully higher
than before (roughly +40% wall-clock in benchmarks) — a real, disclosed tradeoff for
eliminating the discontinuity, not a free change like cross-fitting/shrinkage were.

### Cyclic backfitting

A pairwise interaction's shrunk deviation was the **joint** cell mean — not an
interaction-only signal. If column `a` has a genuine main effect and no real
interaction with `b` exists, the joint `(a, b)` cell mean still reflects `a`'s
main effect (shrunk toward a `dev_a + dev_b` prior that itself contains it), so
the stored pair redundantly re-encodes signal `a`'s own main effect already
captures — and since Lasso stacking can only apply one scalar weight per term,
it can't cleanly cancel a redundant *shape* baked in cell-by-cell. The same gap
applied to triples: the accepted triple's stored value was fit against the raw
residual, even though the accept/reject gain test already computes an
approximate "residual after lower-order terms" for its own threshold decision.

Terms are now fit via a single backfitting pass each round — main effects
first, then pairs (backfit against their own two main effects), then triples
(backfit against their own three main effects, with pairs handled
automatically inside the triple's own recursive prior computation) — so a
pair's or triple's stored value is genuinely interaction-only rather than a
partial copy of what a lower-order term already explains. Not a full
iterate-to-convergence GAM backfit: one ordered pass per round, relying on the
boosting loop's own many rounds for further refinement over time. On data with
a real main effect and no real interaction, this cut the pair term's
Lasso-stacked importance by roughly 40% end-to-end (and by ~5-6x at the level
of a single round's raw fit, before stacking softens it further) — directly
improving what `explain()`/`feature_importance()` show, not just internal
accuracy. Like the four changes above, this is on by default; there's no
tunable knob to expose, so no new parameter was added.

### Monotonic constraints

Unlike the four changes above, this one is **opt-in**: it encodes domain
knowledge the model has no way to infer on its own (e.g. "take-up must
not decrease as affordability rises"), rather than a general correctness
or estimation improvement everyone should get by default. Pass
`monotonic_constraints={"column": +1}` (non-decreasing) or `{"column":
-1}` (non-increasing) — same name/index convention as
`categorical_features` — and that column's own **main effect** is
projected onto the nearest monotonic sequence across its zones via a
row-count-weighted isotonic regression, after empirical Bayes shrinkage
so sparse zones don't distort the projection. Scope is deliberately
narrow:

- **Main effects only.** Pairwise/triple interactions, and the internal
  marginal priors those use for their own shrinkage, are untouched — the
  same scope XGBoost's/LightGBM's own `monotone_constraints` cover.
- A continuous column's zones are already ordered low → high by
  construction, so there's no threshold or window to tune — just a
  direction.
- A constraint declared on a categorical column is silently dropped (no
  meaningful order to constrain for a nominal category) rather than
  raising; an invalid direction (anything other than `-1`/`1`) does
  raise, at `fit()` time.
- The missing-value zone is excluded from the projection — it's a
  separate bucket, not part of the ordered continuum.

Leaving `monotonic_constraints=None` (the default) reproduces the exact
same predictions as before this change — verified bit-for-bit.

### Pair screening

Every round fits **every** `C(p, 2)` pairwise interaction among that
round's (subsampled) predictors — fine for a modest number of columns,
but two costs scale with pair count: cross-fitting recomputes every
pair once per fold (a straight `cross_fit_folds×` multiplier), and Lasso
stacking fits one feature per term, so hundreds/thousands of pairs make
the per-round Lasso fit itself the bottleneck. Like monotonic
constraints, this is **opt-in** — dropping weak pairs entirely changes
results (some would have gotten a small nonzero Lasso weight), so it's
a genuine approximation tradeoff, not a free correctness fix.

`max_pair_interactions` caps how many pairs a round keeps via **cheap-then-
exact hierarchical discovery**, rather than fitting every pair in full and
ranking afterward: every candidate pair is scored with a fast, single-pass
ANOVA-style interaction statistic (does the joint cell mean deviate from
what the two marginals alone would predict) on an honest, cross-fitted
main-effects-only residual — never the same in-sample residual a pair will
later be fit against — and only the top-scoring pairs (plus whatever pairs
the 3-way interaction search needs for its own candidate columns, when
`max_interaction_order=3`) ever pay the full empirical-Bayes fitting cost.
Applied *before* the expensive fit rather than after, so `_select_triples`
still finds genuine 3-way interactions even when only one pair survives
into the final model — its own candidate-column search runs on the cheap
score, computed for every pair, not just the kept ones.

**Measured, honestly**: the per-pair cheap statistic turned out *not* to be
dramatically cheaper than the full fit (roughly 36μs vs. 44μs per pair in
one benchmark) — the real cost driver is the `O(p²)` Python-loop overhead
itself, which both the old and new mechanism pay equally. The net result is
a real but modest **~1.4x** speedup, consistent from 80 to 300 columns, not
an order-of-magnitude win. A fully vectorized screening pass (batching every
pair's joint cell counts via a single matrix multiplication instead of a
Python loop) could close that gap further but isn't implemented here —
noted as a possible future improvement rather than shipped speculatively.
Leaving `max_pair_interactions=None` (the default) keeps every pair — the
exact same behavior as before this change, verified bit-for-bit.

### Prediction intervals (regressor)

`ZoneBoostRegressor.predict_interval(X, alpha=0.1)` returns a constant-width
`(lower, upper)` band around `predict(X)` via **split conformal
prediction** — a distribution-free marginal coverage guarantee,
`P(y in interval) >= 1 - alpha`, assuming exchangeability (Vovk / Lei et
al.'s standard split-conformal setup), not a heteroscedasticity-aware or
locally-adaptive variant. The margin is the finite-sample-corrected
`ceil((n+1)*(1-alpha))`-th smallest absolute residual measured on a
genuinely held-out split — never training rows, so the margin isn't
optimistic about training fit. Purely additive: every existing method's
output is unaffected. Requires `validation_fraction > 0` or
`calibration_fraction > 0` (see "Honest data splits" below); raises
`ValueError` otherwise. On a synthetic noisy quadratic, `alpha=0.1` achieved
~90.2% empirical coverage on held-out data.

### Probability calibration (classifier)

`ZoneBoostClassifier(calibrate=True)` recalibrates each booster's raw
probability with an **isotonic regression** fit on a genuinely held-out
split — the same recipe `sklearn.calibration.CalibratedClassifierCV(
method="isotonic")` uses, so predicted probabilities better match empirical
frequencies. Binary: one calibrator on `booster_`. Multiclass: one per
one-vs-rest booster in `boosters_`, calibrated *before* the existing
cross-class normalization step. On synthetic noisy-sigmoid data, calibration
cut binned reliability error roughly 5x (0.091 → 0.017). Requires
`validation_fraction > 0` or `calibration_fraction > 0`; raises `ValueError`
at `fit` otherwise. Only affects `predict_proba` —
`explain()`/`feature_importance()` still decompose the raw log-odds score
unchanged. This is **opt-in** (default `calibrate=False` reproduces today's
exact `predict_proba` output, verified bit-for-bit) and is the only
parameter that differs between the two estimators.

### Honest data splits (calibration & final refit)

Both calibration mechanisms above originally reused the same
`validation_fraction` split that also drives early stopping — a disclosed
tradeoff (the round count `predict` uses was itself chosen to minimize
error on this exact set, which can understate the true calibration margin
slightly). Two new parameters, shared by both estimators, fix this properly:

- **`calibration_fraction`** (default `0.0`) carves out a **third**,
  genuinely separate partition purely for calibration — never seen by
  either the fit split or the validation split. `0.0` reproduces every
  prior release's behavior exactly (calibration reuses the validation
  split, verified bit-for-bit); setting it removes the disclosed tradeoff
  above entirely.
- **`refit_on_full_data`** (default `False`) — once `best_n_rounds_` is
  chosen from the validation split, optionally retrains the *deployed*
  model on fit+validation data combined, so validation data isn't
  permanently withheld from the model that actually predicts.
  `train_rmse_`/`val_rmse_` still reflect the original selection-phase
  curves, not the refit pass. **Requires `calibration_fraction > 0`**:
  folding the validation split into training means it can no longer double
  as a calibration set too, so a genuinely separate one is required
  (raises `ValueError` otherwise) — this is the one real correctness
  constraint that keeps the two features from silently interacting badly.

Deferred to a future item: cross-conformal/jackknife+ aggregation for small
datasets that can't afford a dedicated calibration split.

## Parameters

Identical parameter set on both estimators, except `calibrate`
(classifier-only — see "Probability calibration" above).

| Parameter | Default | Description |
|---|---|---|
| `n_rounds` | 300 | Maximum number of boosting rounds |
| `learning_rate` | 0.1 | Shrinkage applied to each round's correction |
| `row_subsample` | 0.7 | Fraction of rows sampled per round |
| `col_subsample` | 0.7 | Fraction of columns sampled per round |
| `max_zones` | 7 | Zone cap for *continuous* columns only (see note below) |
| `min_zone_frac` | 0.02 | Minimum row fraction required on each side of a zone split |
| `categorical_features` | None | Column names/indices to treat as nominal categories |
| `validation_fraction` | 0.2 | Held-out fraction used to pick the best round count |
| `n_iter_no_change` | None | Early-stopping patience, in rounds |
| `max_interaction_order` | 2 | Set to 3 to enable an adaptive search for 3-way interactions |
| `max_triple_interactions` | 5 | Cap on how many 3-way terms a single round may add (only relevant when `max_interaction_order=3`) |
| `triple_min_gain` | 0.05 | Minimum residual-explained evidence, relative to a candidate's strongest constituent pair, required to keep a 3-way interaction |
| `cross_fit_folds` | 5 | Number of folds used to compute each round's training signal honestly (see "Cross-fitted cell means" above); falls back to no cross-fitting if a round's row count is smaller than 2 folds |
| `shrinkage_m` | 10.0 | Empirical-Bayes shrinkage strength — a zone needs about this many rows of its own before it's trusted as much as its (hierarchical) prior; see "Empirical Bayes shrinkage" above |
| `stacking_alpha` | 0.01 | Lasso regularization strength for combining a round's terms; see "Lasso stacking" above |
| `monotonic_constraints` | None | `{column: +1 or -1}` — forces a continuous column's main effect to be non-decreasing/non-increasing; opt-in, see "Monotonic constraints" above |
| `max_pair_interactions` | None | Cap on how many pairwise interactions a round keeps, ranked by importance; opt-in, see "Pair screening" above |
| `calibrate` | False | **Classifier only.** Isotonic-recalibrate `predict_proba`; opt-in, see "Probability calibration" above |
| `calibration_fraction` | 0.0 | Fraction held out in a dedicated calibration split, separate from `validation_fraction`; opt-in, see "Honest data splits" above |
| `refit_on_full_data` | False | Refit the deployed model on fit+validation data once `best_n_rounds_` is chosen; requires `calibration_fraction > 0`, see "Honest data splits" above |
| `random_state` | 42 | Seed for the validation split and subsampling |

**On `max_zones` and `categorical_features`:** if a variable genuinely has
many distinct meaningful groups (e.g. a neighborhood or occupation code),
declare it in `categorical_features` rather than raising `max_zones`.
Raising the continuous cap for everyone gives every continuous variable
more per-round fitting flexibility, which in practice mostly helps it
overfit noise rather than capture real structure — proper categorical
handling (exact, uncapped, no ordering assumption) is the fix that's
actually targeted at high-cardinality nominal variables.

## Explaining predictions

Both estimators expose `explain(X)` and `feature_importance(X)`. Unlike
SHAP or LIME, this isn't a post-hoc approximation of a black-box model —
it's an algebraic decomposition of the exact computation `predict`
already performs, so it costs no extra sampling and the result sums
*exactly* to the prediction:

```python
contrib = model.explain(X)            # one column per term, plus "baseline"
contrib.sum(axis=1)                    # == model.predict(X), exactly

model.feature_importance(X)            # mean |contribution| per term, sorted
```

Each column is either a predictor's own name (its main effect), `"A x B"`
(that pair's interaction), or `"A x B x C"` (an adaptively-selected 3-way
interaction, when `max_interaction_order=3`) — never split further, so an
interaction's contribution isn't arbitrarily divided between its
variables. For `ZoneBoostClassifier`, `explain(X)` sums to the **log-odds**
score, not the probability directly (probability contributions don't add
linearly through a sigmoid — the same convention SHAP uses for logistic
models); for 3+ classes it returns a `{class_label: DataFrame}` dict, one
per one-vs-rest booster, and `sigmoid(explain(X)[k].sum(axis=1))`
reproduces that booster's *raw, pre-calibration* probability — equal to
`boosters_[k].predict_proba(X)` only when `calibrate=False` (the default);
with `calibrate=True`, `predict_proba` additionally applies the fitted
isotonic calibrator before the final cross-class normalization.

## Fitted attributes

After `fit`, `ZoneBoostRegressor` exposes (among others):

- `rounds_` — one entry per boosting round, each a plain dict with keys
  `"zone_info"`, `"main_effects"`, `"interactions"`, `"triples"` (empty
  unless `max_interaction_order=3`), and `"intercept"`/`"weights"` — the
  round's fitted Lasso intercept and one weight per term
  (`fitted_residual = intercept + contributions @ weights`, in the same
  order `main_effects`/`interactions`/`triples` are themselves iterated).
  Nothing hidden in an opaque object.
- `best_n_rounds_` — the round count actually used by `predict`.
- `val_rmse_` / `train_rmse_` — RMSE after each round.
- `categorical_features_` — the resolved set of categorical columns
  (declared ∪ auto-detected).
- `monotonic_constraints_` — the resolved `{column: +1 or -1}` dict
  actually in effect (categorical columns dropped).
- `conformal_scores_` — sorted absolute residuals on the held-out
  validation split at `best_n_rounds_`, the nonconformity scores
  `predict_interval` draws its margin from (`None` if
  `validation_fraction=0`); see "Prediction intervals" above.

`ZoneBoostClassifier` exposes the same `categorical_features_`, plus:

- `classes_` — distinct class labels seen during `fit`.
- `multiclass_` — whether one-vs-rest (3+ classes) was used.
- `booster_` (binary) or `boosters_` (a `{class_label: booster}` dict, 3+
  classes) — each an internal log-odds booster with its own `rounds_`,
  `best_n_rounds_`, and `calibrator_` (the fitted isotonic calibrator, or
  `None` if `calibrate=False`) — same plain-data structure as the
  regressor's.

## Benchmarks

Not a leaderboard zoneboost is trying to win — its actual value proposition is
exact, zero-approximation attribution (`explain()`), not necessarily topping
accuracy on tabular benchmarks the way gradient boosting often does. This
reports the real gap (or lack of one) rather than assuming it.

**Regression, California Housing** (3,000-row random subsample, fixed seed,
3-fold cross-validation; each model at its own library's out-of-the-box
defaults — only `random_state` set, no tuning favoring either one):

| Model | RMSE | R² | Fit time (s) | Predict time (s) |
|---|---|---|---|---|
| `ZoneBoostRegressor` | 0.5510 ± 0.0212 | 0.7677 ± 0.0107 | 5.26 | 0.267 |
| `LGBMRegressor` | 0.5221 ± 0.0275 | 0.7912 ± 0.0166 | 1.11 | 0.003 |

LightGBM's RMSE is about 5% lower and it fits roughly 5x faster. zoneboost's
own value here isn't matching or beating LightGBM's raw accuracy — every one
of its predictions decomposes *exactly* into main effects and named
interaction terms via `explain()`, with no sampling and no approximation,
which LightGBM has no built-in equivalent for.

InterpretML's Explainable Boosting Machine (EBM) — architecturally the
closest existing interpretable model — was deliberately left out of this
comparison: its default `outer_bags=8` spawns a separate parallel worker
process per bag per CV fold, and the fixed process-spawn overhead in the
environment this was run in dominated wall-clock time independent of any
real compute cost. That's an environment/parallelism-backend artifact, not a
finding about EBM's accuracy, so it isn't reported here as a misleading
number.

Reproduce with `pip install -e ".[benchmark]"` then
`python benchmarks/compare_lightgbm.py` — see [benchmarks/](benchmarks/) and
the [full write-up](https://stainaz.github.io/zoneboost/benchmarks.html) for
methodology details and how to extend it.

## Scope

This estimator targets practical scikit-learn compatibility —
`get_params`/`set_params`/`clone`, use inside a `Pipeline`, and scoring via
`cross_val_score` — rather than full compliance with
`sklearn.utils.estimator_checks.check_estimator`, which checks many edge
cases (e.g. sparse-matrix input) not exercised here.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
