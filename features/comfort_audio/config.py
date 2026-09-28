"""Configuration dataclass for Adaptive Auditory Comfort Layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ComfortAudioConfig:
    enabled: bool = False
    mode: Literal["pink_noise", "neutral_tone", "ambient_file"] = "pink_noise"
    volume: float = 0.15
    speech_threshold_db: float = -42.0
    fade_out_ms: float = 60.0
    hold_ms: float = 400.0
    fade_in_ms: float = 1000.0
    tone_freq_hz: float = 120.0
    ambient_file_path: str | None = None
    communication_priority: bool = True
    transient_ducking: bool = True

    def validate(self) -> None:
        if not (0.0 <= self.volume <= 1.0):
            raise ValueError(f"Volume must be in [0.0, 1.0], got {self.volume}")
        if self.fade_out_ms <= 0 or self.fade_in_ms <= 0:
            raise ValueError("Fade times must be positive numbers")
        if self.mode not in ("pink_noise", "neutral_tone", "ambient_file"):
            raise ValueError(f"Unknown mode: {self.mode}")
