"""Tests for the guard that refuses to start on a leaky configuration."""

import importlib

import pytest


def _config(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import hearthvoice.config as config
    return importlib.reload(config)


def test_local_configuration_passes(monkeypatch):
    config = _config(monkeypatch, SPARK_VIP="192.168.60.49",
                     LLM_BASE_URL="http://192.168.60.49:19080/v1",
                     NVIDIA_STT_SERVER="192.168.60.49:50051",
                     NVIDIA_TTS_SERVER="192.168.60.49:50053",
                     MQTT_HOST="127.0.0.1")
    assert len(config.check_offline()) == 5  # four stages plus Home Assistant


def test_a_vendor_endpoint_is_refused(monkeypatch):
    config = _config(monkeypatch, SPARK_VIP="192.168.60.49",
                     LLM_BASE_URL="https://api.openai.com/v1",
                     NVIDIA_STT_SERVER="192.168.60.49:50051",
                     NVIDIA_TTS_SERVER="192.168.60.49:50053",
                     MQTT_HOST="127.0.0.1")
    with pytest.raises(config.OfflineViolation, match="api.openai.com"):
        config.check_offline()


def test_each_stage_is_checked_separately(monkeypatch):
    """A leak in speech synthesis is caught even when the rest is local."""
    config = _config(monkeypatch, SPARK_VIP="192.168.60.49",
                     LLM_BASE_URL="http://192.168.60.49:19080/v1",
                     NVIDIA_STT_SERVER="192.168.60.49:50051",
                     NVIDIA_TTS_SERVER="tts.example.com:50053",
                     MQTT_HOST="127.0.0.1")
    with pytest.raises(config.OfflineViolation, match="TTS"):
        config.check_offline()


def test_the_guard_cannot_be_talked_round(monkeypatch):
    """Pointing the allowlist itself at a vendor must not make it agree.

    The first version built its allowlist out of SPARK_VIP, the same variable
    it was checking, so setting that to a vendor host made every stage pass. A
    check that can be satisfied by changing its own input is not a check.
    """
    config = _config(monkeypatch, SPARK_VIP="api.openai.com",
                     LLM_BASE_URL="https://api.openai.com/v1",
                     NVIDIA_STT_SERVER="api.openai.com:50051",
                     NVIDIA_TTS_SERVER="api.openai.com:50053",
                     MQTT_HOST="127.0.0.1", HA_URL="http://127.0.0.1:8123")
    with pytest.raises(config.OfflineViolation):
        config.check_offline()


def test_home_assistant_is_checked_too(monkeypatch):
    """It was left out of the first version entirely."""
    config = _config(monkeypatch, SPARK_VIP="192.168.60.49",
                     LLM_BASE_URL="http://192.168.60.49:19080/v1",
                     NVIDIA_STT_SERVER="192.168.60.49:50051",
                     NVIDIA_TTS_SERVER="192.168.60.49:50053",
                     MQTT_HOST="127.0.0.1",
                     HA_URL="http://someone-elses-server.example.com:8123")
    with pytest.raises(config.OfflineViolation, match="Home Assistant"):
        config.check_offline()


def test_a_public_address_is_refused_even_if_it_looks_like_a_server(monkeypatch):
    config = _config(monkeypatch, SPARK_VIP="8.8.8.8",
                     LLM_BASE_URL="http://8.8.8.8:19080/v1",
                     NVIDIA_STT_SERVER="8.8.8.8:50051",
                     NVIDIA_TTS_SERVER="8.8.8.8:50053",
                     MQTT_HOST="127.0.0.1", HA_URL="http://127.0.0.1:8123")
    with pytest.raises(config.OfflineViolation):
        config.check_offline()
