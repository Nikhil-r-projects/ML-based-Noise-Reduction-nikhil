"""End-to-end integration test for Extended Defence ANC System."""
import numpy as np
from features.comfort_audio import ComfortAudioConfig, ComfortAudioMixer
from features.speech_to_text import ASRConfig, get_asr_engine
from features.situational_awareness import RadioProtocolParser, EventBus


def test_end_to_end_audio_to_tactical_event_pipeline():
    sample_rate = 16000
    # 1. Simulate enhanced speech from ANC-Net v3 (1-channel 16 kHz)
    enhanced_speech = np.random.uniform(-0.4, 0.4, size=sample_rate * 2).astype(np.float32)

    # 2. Comfort Audio Layer (mixes pink noise bed with ducking)
    comfort_cfg = ComfortAudioConfig(enabled=True, volume=0.15, mode="pink_noise")
    mixer = ComfortAudioMixer(comfort_cfg)
    mixed_audio, envelope = mixer.mix(enhanced_speech, sample_rate=sample_rate)

    assert len(mixed_audio) == len(enhanced_speech)
    assert len(envelope) == len(enhanced_speech)

    # 3. Asynchronous ASR transcription
    asr_cfg = ASRConfig(backend="mock")
    asr_engine = get_asr_engine(asr_cfg)
    transcript = asr_engine.transcribe(enhanced_speech, sample_rate=sample_rate)

    assert transcript.text != ""
    assert transcript.timestamp != ""

    # 4. Deterministic Tactical Entity Extraction
    parser = RadioProtocolParser()
    event = parser.parse(transcript.text, timestamp=transcript.timestamp)

    assert event.entity != ""
    assert event.validate() is True

    # 5. Broadcast to Event Bus (for Web/Map UI)
    bus = EventBus()
    bus.publish(event)

    history = bus.get_history()
    assert len(history) == 1
    assert history[0]["entity"] == event.entity
    assert history[0]["raw_transcript"] == transcript.text
