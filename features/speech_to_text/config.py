"""Configuration for Speech-to-Text (ASR) module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ASRConfig:
    backend: Literal["faster_whisper", "onnx", "mock"] = "faster_whisper"
    model_size: str = "tiny.en"
    device: str = "cpu"
    compute_type: str = "int8"
    sample_rate: int = 16000
    queue_maxsize: int = 16
    language: str = "en"
    beam_size: int = 1
