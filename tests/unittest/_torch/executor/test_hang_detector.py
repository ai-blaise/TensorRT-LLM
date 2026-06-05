import pytest

from tensorrt_llm._torch.pyexecutor.hang_detector import (
    PYEXECUTOR_HANG_DETECTION_TIMEOUT_ENV,
    HangDetector,
)


def test_hang_detector_uses_default_timeout(monkeypatch):
    monkeypatch.delenv(PYEXECUTOR_HANG_DETECTION_TIMEOUT_ENV, raising=False)

    assert HangDetector().timeout == 300


def test_hang_detector_uses_env_timeout(monkeypatch):
    monkeypatch.setenv(PYEXECUTOR_HANG_DETECTION_TIMEOUT_ENV, "1200")

    assert HangDetector().timeout == 1200


def test_hang_detector_explicit_timeout_overrides_env(monkeypatch):
    monkeypatch.setenv(PYEXECUTOR_HANG_DETECTION_TIMEOUT_ENV, "1200")

    assert HangDetector(timeout=30).timeout == 30


def test_hang_detector_rejects_non_integer_env_timeout(monkeypatch):
    monkeypatch.setenv(PYEXECUTOR_HANG_DETECTION_TIMEOUT_ENV, "slow")

    with pytest.raises(ValueError, match=PYEXECUTOR_HANG_DETECTION_TIMEOUT_ENV):
        HangDetector()
