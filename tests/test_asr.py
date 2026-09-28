"""Tests for Speech-to-Text module."""
import time
import numpy as np
from features.speech_to_text import ASRConfig, get_asr_engine, ASRWorkerThread


def test_mock_asr_engine_transcription():
    cfg = ASRConfig(backend="mock")
    engine = get_asr_engine(cfg)

    # Audio with active signal
    active_audio = np.random.uniform(-0.3, 0.3, size=16000).astype(np.float32)
    result = engine.transcribe(active_audio, sample_rate=16000)

    assert result.text != ""
    assert result.duration_s == 1.0
    assert result.confidence is not None
    assert result.source == "enhanced_audio"


def test_mock_asr_silence_handling():
    cfg = ASRConfig(backend="mock")
    engine = get_asr_engine(cfg)

    silence = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(silence, sample_rate=16000)

    assert result.text == ""


def test_asr_worker_thread_asynchronous():
    received_results = []

    def on_transcript(res):
        received_results.append(res)

    cfg = ASRConfig(backend="mock")
    worker = ASRWorkerThread(config=cfg, on_transcript=on_transcript)
    worker.start()

    try:
        audio_chunk = np.random.uniform(-0.2, 0.2, size=16000).astype(np.float32)
        success = worker.submit_audio(audio_chunk)
        assert success is True

        # Wait briefly for worker thread execution
        time.sleep(0.4)
        assert len(received_results) >= 1
        assert received_results[0].text != ""
    finally:
        worker.stop()
