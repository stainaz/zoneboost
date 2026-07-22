"""LLM-assisted zone naming: a business-language label for a zone or
zone-pair ("young, low-affordability, high-claims corridor") in place of a
table of cut points -- so an audit artifact reads like an underwriting
manual instead of a list of numeric ranges.

This is the first zoneboost feature needing a dependency beyond numpy/
pandas/scikit-learn, so it's gated behind the ``zoneboost[llm]`` extra and
never imported at module load time: ``import zoneboost`` (and even
``from zoneboost import LLMZoneNamer``) works with zero extra dependencies
installed -- only calling ``LLMZoneNamer(...).name_zones(...)`` with no
injected ``client`` requires ``anthropic`` to actually be present.

Deliberately decoupled from every other zoneboost internal: ``name_zones``
takes plain zone descriptions the caller already has (from
:attr:`zoneboost.ZoneProfileEncoder.zone_stats_`,
:attr:`zoneboost.ConditionalZoneGrid.segment_grids_`, or built by hand),
not a fitted model or its ``rounds_`` -- the same compose-rather-than-
couple precedent :class:`zoneboost.ZoneProfileEncoder`'s own pairwise-
profiling "Deferred" note set.
"""

from __future__ import annotations

import json

from sklearn.base import BaseEstimator

__all__ = ["LLMZoneNamer"]

_NAMES_SCHEMA = {
    "type": "object",
    "properties": {
        "names": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["names"],
    "additionalProperties": False,
}


def _build_prompt(zone_summaries: list, context: str = None) -> str:
    lines = []
    if context:
        lines.append(f"Business context: {context}")
    lines.append(
        "For each zone described below, write a short (3-6 word) business-language "
        "name that captures what distinguishes it -- the kind of label an underwriter "
        "or analyst would recognize, not a restatement of the raw numbers. Return "
        "exactly one name per zone, in the same order."
    )
    lines.append("Zones:")
    for i, summary in enumerate(zone_summaries):
        lines.append(f"{i}: {json.dumps(summary, default=str)}")
    return "\n".join(lines)


def _parse_response(response, expected: int) -> list:
    text = next(block.text for block in response.content if block.type == "text")
    names = json.loads(text)["names"]
    if len(names) != expected:
        raise ValueError(
            f"Expected {expected} zone names back from the model, got {len(names)}. "
            "Refusing to guess which name belongs to which zone."
        )
    return names


class LLMZoneNamer(BaseEstimator):
    """Business-language names for a batch of zone descriptions, via the
    Claude API.

    Requires the optional ``anthropic`` package (``pip install
    zoneboost[llm]``) unless a ``client`` is supplied directly.

    Parameters
    ----------
    client : object, default=None
        Anything exposing ``.messages.create(...)`` with the Anthropic
        Messages API's response shape. ``None`` (default) lazily
        constructs a bare ``anthropic.Anthropic()`` on first use, which
        resolves credentials from the environment (``ANTHROPIC_API_KEY``,
        or an ``ant auth login`` profile) -- no key handling of zoneboost's
        own. Passing a fake/mock client (e.g. in tests) never touches the
        network or requires ``anthropic`` to be installed at all.
    model : str, default="claude-opus-4-8"
        Model ID passed to ``messages.create``.
    max_tokens : int, default=1024
        Passed to ``messages.create``. Scale up for large batches of zones.

    Examples
    --------
    >>> from zoneboost import LLMZoneNamer
    >>> zones = [
    ...     {"feature": "age", "range": (18, 25), "count": 812, "outcome_rate": 0.31},
    ...     {"feature": "age", "range": (45, 60), "count": 1204, "outcome_rate": 0.06},
    ... ]
    >>> namer = LLMZoneNamer()
    >>> names = namer.name_zones(zones, context="auto insurance underwriting")  # doctest: +SKIP
    """

    def __init__(self, client=None, model: str = "claude-opus-4-8", max_tokens: int = 1024):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def _default_client(self):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "LLMZoneNamer requires the 'anthropic' package. Install it with "
                "`pip install zoneboost[llm]`, or pass your own `client`."
            ) from e
        return anthropic.Anthropic()

    def name_zones(self, zone_summaries, context: str = None) -> list:
        """Name every zone in ``zone_summaries``, in order.

        Parameters
        ----------
        zone_summaries : list of dict
            One plain dict per zone -- whatever fields describe it (e.g.
            ``feature``, ``range`` or ``category``, ``count``,
            ``outcome_rate``/``mean``). Built by the caller from
            ``ZoneProfileEncoder.zone_stats_``, ``ConditionalZoneGrid.
            segment_grids_``, or by hand -- this method doesn't parse
            zoneboost's own internal shapes directly.
        context : str, default=None
            One-line business context prepended to the prompt (e.g.
            "auto insurance underwriting"), to steer naming vocabulary.

        Returns
        -------
        list of str
            One name per input zone, same order. Raises ``ValueError`` if
            the model returns a different count than requested, rather
            than guessing which name belongs to which zone.
        """
        client = self.client if self.client is not None else self._default_client()
        prompt = _build_prompt(zone_summaries, context)
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": _NAMES_SCHEMA}},
        )
        return _parse_response(response, expected=len(zone_summaries))
