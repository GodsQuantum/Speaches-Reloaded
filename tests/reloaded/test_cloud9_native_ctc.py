"""Cloud9 native CTC regression: reject ungrounded timings and preserve Han tokens."""

import json
from pathlib import Path

import pytest

from speaches.executors.cloud9_alignment_groups import groups_for_alignment
from speaches.executors.cloud9_ctc import acoustic_tokens, restore_original_tokens, selected_compute_backend
from speaches.executors.cloud9_word_alignment import spoken_tokens


def test_chinese_acoustic_tokens_not_single_utterance() -> None:
    assert acoustic_tokens("你好世界。 OpenAI") == ["你", "好", "世", "界。", "OpenAI"]
    assert spoken_tokens("你好世界。 OpenAI") == ["你", "好", "世", "界。", "OpenAI"]


def test_longform_requires_real_asr_chunk_offsets() -> None:
    with pytest.raises(ValueError, match="long audio requires"):
        groups_for_alignment("bonjour à toutes et à tous", 104.96, None)


def test_mismatched_transcript_rejected() -> None:
    rows = [{"start": 0, "end": 8, "text": "Bonjour les amis"}]
    with pytest.raises(ValueError, match="differ"):
        groups_for_alignment("Bonsoir les amis", 8.0, json.dumps(rows))


def test_hyphen_preserves_measured_start_end() -> None:
    raw = [{"word": "Tais-", "start": 0.10, "end": 0.36}, {"word": "toi", "start": 0.38, "end": 0.72}]
    assert restore_original_tokens(raw, "Tais-toi") == [{"word": "Tais-toi", "start": 0.10, "end": 0.72}]


def test_float32_wav_is_losslessly_timed_as_pcm16(tmp_path: Path) -> None:
    """One valid input frame must remain one acoustic frame after conversion."""
    import numpy as np
    import soundfile as sf

    from speaches.executors.cloud9_ctc import pcm16_wave

    original = tmp_path / "float32.wav"
    samples = np.linspace(-0.8, 0.8, 16000, dtype=np.float32)
    sf.write(str(original), samples, 16000, subtype="FLOAT")
    with pcm16_wave(original) as reader:
        assert reader.getframerate() == 16000
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getnframes() == 16000
        assert len(reader.readframes(16000)) == 32000


def test_pcm16_input_retains_existing_wave_path(tmp_path: Path) -> None:
    import wave

    from speaches.executors.cloud9_ctc import pcm16_wave

    path = tmp_path / "pcm16.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 16000)
    with pcm16_wave(path) as reader:
        assert reader.getnframes() == 16000
        assert reader.readframes(1) == b"\x00\x00"


def test_trailing_closing_guillemet_is_not_a_spoken_word() -> None:
    raw = [
        {"word": "Bonjour", "start": 0.1, "end": 0.5},
        {"word": "»", "start": 0.51, "end": 0.75},
    ]
    assert restore_original_tokens(raw, "Bonjour»") == [{"word": "Bonjour»", "start": 0.1, "end": 0.5}]


def test_trailing_real_word_still_rejected() -> None:
    raw = [
        {"word": "Bonjour", "start": 0.1, "end": 0.5},
        {"word": "encore", "start": 0.51, "end": 0.75},
    ]
    with pytest.raises(ValueError, match="extra words"):
        restore_original_tokens(raw, "Bonjour")


def test_trailing_nonpunctuation_mark_still_rejected() -> None:
    raw = [
        {"word": "Bonjour", "start": 0.1, "end": 0.5},
        {"word": " ", "start": 0.51, "end": 0.75},
    ]
    with pytest.raises(ValueError, match="extra words"):
        restore_original_tokens(raw, "Bonjour")


def test_missing_acoustic_word_must_fail() -> None:
    with pytest.raises(ValueError, match=r"omitted|mismatch"):
        restore_original_tokens([{"word": "Bonjour", "start": 0.1, "end": 0.5}], "Bonjour les amis")


@pytest.mark.parametrize("backend", ["auto", "cpu", "cuda", "vulkan", "metal"])
def test_ctc_backend_selection_all_platforms(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    monkeypatch.setenv("CLOUD9_CTC_BACKEND", backend.upper())
    assert selected_compute_backend() == backend


def test_ctc_backend_rejects_unsupported_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD9_CTC_BACKEND", "vulkan-and-guess")
    with pytest.raises(ValueError, match="Invalid CLOUD9_CTC_BACKEND"):
        selected_compute_backend()


def test_native_ctc_deploy_recipe_copies_full_merged_stack() -> None:
    """A successful source PR must not silently disappear from the Docker image."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    dockerfile = (repo / "deploy/cloud9/Dockerfile.native-ctc").read_text(encoding="utf8")
    required_sources = (
        "src/speaches/config.py",
        "src/speaches/executors/cloud9_alignment_groups.py",
        "src/speaches/executors/cloud9_longform.py",
        "src/speaches/executors/cloud9_ctc.py",
        "src/speaches/executors/cloud9_word_alignment.py",
        "src/speaches/ui/tabs/alignment.py",
    )
    for source in required_sources:
        assert f"COPY --chown=ubuntu:ubuntu {source} {source}" in dockerfile


def test_native_ctc_deployment_guide_keeps_reversible_image_upgrade() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    guide = (repo / "docs/cloud9-native-vulkan-ctc.md").read_text(encoding="utf8")
    assert "## Rollback" in guide
    assert "docker compose -f /etc/speaches/compose.yaml" in guide
    assert "cloud9-protected.slice" in guide
    assert "CUDA and Metal binaries / real hardware throughput are **not certified**" in guide
