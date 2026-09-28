"""Whisper and fallback backends for local speech recognition."""
from __future__ import annotations

import datetime
import math
import numpy as np
from .asr_engine import ASREngine
from .config import ASRConfig
from .models import TranscriptResult


class FasterWhisperBackend(ASREngine):
    """Local ASR inference engine backed by faster-whisper (CTranslate2)."""

    def __init__(self, config: ASRConfig | None = None):
        super().__init__(config)
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            self.cfg.model_size,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptResult:
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        duration_s = len(audio) / sample_rate
        if len(audio) == 0:
            return TranscriptResult(text="", timestamp=now_str, duration_s=0.0, confidence=1.0)

        # faster-whisper expects 16 kHz float32 in [-1, 1]
        segments, info = self.model.transcribe(
            audio,
            beam_size=self.cfg.beam_size,
            language=self.cfg.language,
        )
        text_parts = []
        confidences = []
        for segment in segments:
            text_parts.append(segment.text.strip())
            if segment.avg_logprob is not None:
                conf = math.exp(max(-10.0, segment.avg_logprob))
                confidences.append(conf)

        full_text = " ".join(text_parts).strip()
        mean_conf = float(np.mean(confidences)) if confidences else None

        return TranscriptResult(
            text=full_text,
            timestamp=now_str,
            duration_s=round(duration_s, 2),
            confidence=round(mean_conf, 3) if mean_conf is not None else None,
            source="enhanced_audio",
        )


class MockASRBackend(ASREngine):
    """Deterministic ASR backend for testing and edge fallbacks."""

    SAMPLE_RADIO_PHRASES = [
        "Alpha team moving towards checkpoint Bravo.",
        "Charlie team reached sector four.",
        "Bravo team waiting at checkpoint two.",
        "Delta squad under fire at objective Iron.",
        "Echo element holding position at sector seven.",
        "Viper recon requesting evac at landing zone Alpha.",
    ]

    def __init__(self, config: ASRConfig | None = None):
        super().__init__(config)
        self.phrase_index = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptResult:
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        duration_s = round(len(audio) / max(1, sample_rate), 2)
        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
        
        # If signal is near silence, return empty transcript
        if rms < 0.005 or len(audio) < sample_rate * 0.5:
            return TranscriptResult(
                text="",
                timestamp=now_str,
                duration_s=duration_s,
                confidence=0.99,
                source="enhanced_audio",
            )

        # Cycle through representative tactical radio communications
        phrase = self.SAMPLE_RADIO_PHRASES[self.phrase_index % len(self.SAMPLE_RADIO_PHRASES)]
        self.phrase_index += 1

        return TranscriptResult(
            text=phrase,
            timestamp=now_str,
            duration_s=duration_s,
            confidence=0.92,
            source="enhanced_audio",
        )
