"""zoneboost.eda -- the visual, business-friendly EDA layer the notebook
pages asked for ("compare outcome & variables boxplots", "direction
included") on top of zoneboost's own zone construction, drift comparison,
and explain().

A separate subpackage, not part of the top-level ``zoneboost`` namespace:
``matplotlib`` is a new dependency beyond ``numpy``/``pandas``/
``scikit-learn`` (gated behind ``pip install zoneboost[eda]``, the same
optional-extra precedent :class:`zoneboost.LLMZoneNamer` set for
``anthropic``), and every function here still returns its underlying
numbers as a plain DataFrame/dict even when a plot is drawn -- nothing is
only inspectable as a rendered image. ``import zoneboost`` and even
``from zoneboost.eda import zone_boxplot`` work with zero extra
dependencies installed; only ``plot=True`` (the default) reaches the
``matplotlib`` import.
"""

from __future__ import annotations

from ._drift_dashboard import drift_dashboard
from ._prediction_waterfall import prediction_waterfall
from ._signed_contribution_profile import signed_contribution_profile
from ._zone_boxplot import zone_boxplot

__all__ = ["zone_boxplot", "drift_dashboard", "prediction_waterfall", "signed_contribution_profile"]
