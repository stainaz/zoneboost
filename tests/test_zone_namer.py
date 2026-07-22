import builtins
import json
import sys

import pytest
from sklearn.base import clone

from zoneboost import LLMZoneNamer


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, names):
        self.content = [_FakeTextBlock(json.dumps({"names": names}))]


class _FakeMessages:
    def __init__(self, names):
        self._names = names
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self._names)


class _FakeClient:
    def __init__(self, names):
        self.messages = _FakeMessages(names)


_ZONES = [
    {"feature": "age", "range": (18, 25), "count": 812, "outcome_rate": 0.31},
    {"feature": "age", "range": (45, 60), "count": 1204, "outcome_rate": 0.06},
]


def test_name_zones_returns_names_in_order():
    client = _FakeClient(["Young high-risk", "Established low-risk"])
    namer = LLMZoneNamer(client=client)
    names = namer.name_zones(_ZONES)
    assert names == ["Young high-risk", "Established low-risk"]


def test_prompt_includes_every_zone_and_context():
    client = _FakeClient(["A", "B"])
    namer = LLMZoneNamer(client=client)
    namer.name_zones(_ZONES, context="auto insurance underwriting")
    prompt = client.messages.last_kwargs["messages"][0]["content"]
    assert "auto insurance underwriting" in prompt
    assert "18" in prompt and "25" in prompt
    assert "45" in prompt and "60" in prompt


def test_uses_configured_model_and_max_tokens():
    client = _FakeClient(["A", "B"])
    namer = LLMZoneNamer(client=client, model="claude-opus-4-8", max_tokens=512)
    namer.name_zones(_ZONES)
    assert client.messages.last_kwargs["model"] == "claude-opus-4-8"
    assert client.messages.last_kwargs["max_tokens"] == 512


def test_uses_structured_output_schema():
    client = _FakeClient(["A", "B"])
    namer = LLMZoneNamer(client=client)
    namer.name_zones(_ZONES)
    fmt = client.messages.last_kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["required"] == ["names"]


def test_mismatched_name_count_raises():
    client = _FakeClient(["Only one name"])
    namer = LLMZoneNamer(client=client)
    with pytest.raises(ValueError):
        namer.name_zones(_ZONES)


def test_default_model_is_claude_opus_4_8():
    namer = LLMZoneNamer()
    assert namer.model == "claude-opus-4-8"


def test_missing_anthropic_package_raises_import_error_with_install_hint(monkeypatch):
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)

    namer = LLMZoneNamer()
    with pytest.raises(ImportError, match=r"zoneboost\[llm\]"):
        namer.name_zones(_ZONES)


def test_injected_client_is_used_without_constructing_a_default(monkeypatch):
    def _fail_if_called(name, *args, **kwargs):
        if name == "anthropic":
            raise AssertionError("should not import anthropic when a client is injected")
        return builtins.__import__(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_if_called)
    client = _FakeClient(["A", "B"])
    namer = LLMZoneNamer(client=client)
    names = namer.name_zones(_ZONES)
    assert names == ["A", "B"]


def test_get_params_and_clone_work():
    model = LLMZoneNamer(model="claude-opus-4-8", max_tokens=2048)
    params = model.get_params()
    assert params["max_tokens"] == 2048

    cloned = clone(model)
    assert cloned.max_tokens == 2048
    assert cloned.model == "claude-opus-4-8"
    assert cloned is not model
