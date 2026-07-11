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

## Parameters

Identical parameter set on both estimators.

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
reproduces that booster's *raw* probability before the final
cross-class normalization `predict_proba` applies.

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

`ZoneBoostClassifier` exposes the same `categorical_features_`, plus:

- `classes_` — distinct class labels seen during `fit`.
- `multiclass_` — whether one-vs-rest (3+ classes) was used.
- `booster_` (binary) or `boosters_` (a `{class_label: booster}` dict, 3+
  classes) — each an internal log-odds booster with its own `rounds_` and
  `best_n_rounds_`, same plain-data structure as the regressor's.

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
