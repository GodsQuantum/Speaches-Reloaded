"""Regression: current and legacy Sortformer Python binding APIs."""

from __future__ import annotations

from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace

import pytest

from speaches.executors.transcribe_cpp import TranscribeCppDiarizationModelManager


def _setup(monkeypatch: pytest.MonkeyPatch, binding: ModuleType, model: object):
    monkeypatch.setitem(sys.modules, "transcribe_cpp", binding)
    manager = TranscribeCppDiarizationModelManager(ttl=20)
    monkeypatch.setattr(manager, "load_model", lambda _id: nullcontext(model))
    return manager, SimpleNamespace(data=[0.0, 0.1])


def test_current_sortformer_diarization_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    turns = [SimpleNamespace(speaker_id=1, t0_ms=0, t1_ms=200)]
    binding = ModuleType("transcribe_cpp")
    binding.SortformerDiarizeOptions = lambda preset: ("new", preset)

    def run(audio, *, family):
        calls.append((audio, family))
        return turns

    model = SimpleNamespace(
        diarize_session=lambda: nullcontext(SimpleNamespace(run=run))
    )
    manager, audio = _setup(monkeypatch, binding, model)
    result = manager.diarize("model", audio)

    assert result.speaker_segments == tuple(turns)
    assert calls == [(audio.data, ("new", "very_high_latency"))]


def test_legacy_sortformer_api_still_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    binding = ModuleType("transcribe_cpp")
    binding.SortformerStreamOptions = lambda preset: ("old", preset)
    expected = SimpleNamespace(speaker_segments=["legacy"])

    def run(audio, *, timestamps, family):
        calls.append((audio, timestamps, family))
        return expected

    model = SimpleNamespace(session=lambda: nullcontext(SimpleNamespace(run=run)))
    manager, audio = _setup(monkeypatch, binding, model)
    assert manager.diarize("model", audio) is expected
    assert calls == [(audio.data, "none", ("old", "very_high_latency"))]


def test_unsupported_sortformer_api_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = ModuleType("transcribe_cpp")
    manager, audio = _setup(monkeypatch, binding, SimpleNamespace())
    with pytest.raises(RuntimeError, match="supported Sortformer diarization API"):
        manager.diarize("model", audio)
