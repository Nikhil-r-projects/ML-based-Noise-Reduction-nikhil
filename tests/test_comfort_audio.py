"""Tests for Adaptive Auditory Comfort Layer."""
import numpy as np
import pytest
from features.comfort_audio import ComfortAudioConfig, ComfortAudioController, ComfortAudioMixer


def test_comfort_audio_config_validation():
    cfg = ComfortAudioConfig(volume=0.5, fade_in_ms=500.0, mode="pink_noise")
    cfg.validate()

    with pytest.raises(ValueError):
        ComfortAudioConfig(volume=1.5).validate()

    with pytest.raises(ValueError):
        ComfortAudioConfig(fade_out_ms=-10.0).validate()

    with pytest.raises(ValueError):
        ComfortAudioConfig(mode="unsupported_mode").validate()  # type: ignore


def test_comfort_audio_disabled():
    cfg = ComfortAudioConfig(enabled=False, volume=0.5)
    mixer = ComfortAudioMixer(cfg)
    speech = np.random.uniform(-0.1, 0.1, size=1600).astype(np.float32)

    mixed, envelope = mixer.mix(speech, sample_rate=16000)
    assert np.allclose(mixed, speech)
    assert np.all(envelope == 0.0)


def test_comfort_audio_ducking_on_loud_speech():
    # Configure fast fade-out and high volume
    cfg = ComfortAudioConfig(
        enabled=True,
        volume=0.3,
        speech_threshold_db=-40.0,
        fade_out_ms=10.0,
        hold_ms=50.0,
        fade_in_ms=100.0,
    )
    controller = ComfortAudioController(cfg)
    sample_rate = 16000

    # 1. Loud speech frame (active speech, e.g. amplitude 0.5 ~ -6 dBFS)
    loud_speech = np.full(320, 0.5, dtype=np.float32)
    env_ducking = controller.process_frame(loud_speech, sample_rate)

    # Envelope should decrease towards 0
    assert env_ducking[-1] < env_ducking[0]
    assert env_ducking[-1] < 0.05


def test_comfort_audio_transient_instant_mute():
    cfg = ComfortAudioConfig(enabled=True, volume=0.3, transient_ducking=True)
    controller = ComfortAudioController(cfg)
    sample_rate = 16000

    speech = np.zeros(160, dtype=np.float32)
    # When a gunshot/blast transient occurs, envelope must be forced to 0 immediately
    env = controller.process_frame(speech, sample_rate, has_transient=True)
    assert np.all(env == 0.0)


def test_comfort_audio_synthesis_modes():
    for mode in ["pink_noise", "neutral_tone"]:
        cfg = ComfortAudioConfig(enabled=True, volume=0.2, mode=mode)  # type: ignore
        mixer = ComfortAudioMixer(cfg)
        bed = mixer.generate_bed(1600, sample_rate=16000)
        assert len(bed) == 1600
        assert np.max(np.abs(bed)) > 0.0
        assert np.max(np.abs(bed)) <= 1.0
