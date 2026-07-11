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

Every zone's contribution is weighted by **density confidence** — its row
count relative to the best-supported zone that round — so sparse zones
contribute less than well-supported ones. Each round's correction is
applied at a small, shrunk step (`learning_rate`) and added to a running
prediction, exactly like standard gradient boosting. `row_subsample` /
`col_subsample` add stochastic-gradient-boosting-style regularization by
fitting each round on a random subsample of rows and columns.

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
effects, same interactions, same density confidence. The only change is
where boosting happens: each round is fit against the residual in
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
  unless `max_interaction_order=3`), and that round's rescaling stats.
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
