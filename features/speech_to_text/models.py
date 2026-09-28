"""Data models for Speech-to-Text module."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class TranscriptResult:
    text: str
    timestamp: str
    duration_s: float
    confidence: float | None = None
    source: str = "enhanced_audio"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
