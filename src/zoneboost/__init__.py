"""zoneboost -- fully transparent, zone-based gradient boosting.

No decision trees, no gradient descent, no neural weights. Every number in
a prediction traces back to a quantile, a group count, or a group average.

    >>> from zoneboost import ZoneBoostRegressor
    >>> model = ZoneBoostRegressor().fit(X_train, y_train)
    >>> model.predict(X_test)

    >>> from zoneboost import ZoneBoostClassifier
    >>> model = ZoneBoostClassifier().fit(X_train, y_train)
    >>> model.predict_proba(X_test)

See :class:`ZoneBoostRegressor` / :class:`ZoneBoostClassifier` for the full
parameter and attribute reference.
"""

from ._bootstrap import BootstrapStability
from ._conditional_grid import ConditionalZoneGrid
from ._conformal import ConformalizedQuantileRegressor
from ._depth import DepthTransformer
from ._drift import compare_models
from ._sql_export import compile_to_sql
from ._survival import ZoneBoostSurvival
from ._version import __version__
from ._zone_profile import ZoneProfileEncoder
from .classifier import ZoneBoostClassifier
from .regressor import ZoneBoostRegressor

__all__ = [
    "ZoneBoostRegressor",
    "ZoneBoostClassifier",
    "ConformalizedQuantileRegressor",
    "BootstrapStability",
    "ZoneBoostSurvival",
    "ZoneProfileEncoder",
    "DepthTransformer",
    "ConditionalZoneGrid",
    "compare_models",
    "compile_to_sql",
    "__version__",
]
