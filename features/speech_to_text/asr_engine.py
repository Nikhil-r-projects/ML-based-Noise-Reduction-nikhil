"""Abstract Base Class and factory for Speech-to-Text engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from .config import ASRConfig
from .models import TranscriptResult


class ASREngine(ABC):
    """Abstract interface for local speech-to-text inference engines."""

    def __init__(self, config: ASRConfig | None = None):
        self.cfg = config or ASRConfig()

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptResult:
        """Transcribe 1D float32 audio array into a structured TranscriptResult."""
        pass


def get_asr_engine(config: ASRConfig | None = None) -> ASREngine:
    """Factory creating the appropriate ASR engine based on configuration."""
    cfg = config or ASRConfig()
    if cfg.backend == "faster_whisper":
        try:
            from .whisper_backend import FasterWhisperBackend
            return FasterWhisperBackend(cfg)
        except (ImportError, Exception):
            from .whisper_backend import MockASRBackend
            return MockASRBackend(cfg)
    elif cfg.backend == "mock":
        from .whisper_backend import MockASRBackend
        return MockASRBackend(cfg)
    else:
        from .whisper_backend import MockASRBackend
        return MockASRBackend(cfg)
