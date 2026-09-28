"""Data schemas and validators for Situational Awareness events."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal


@dataclass
class GridCoordinates:
    x: int
    y: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass
class TacticalEvent:
    entity: str
    event_type: Literal["movement", "arrival", "holding", "contact", "evacuation", "unknown"]
    unit_type: str = "infantry"
    origin: str | None = None
    destination: str | None = None
    location: str | None = None
    coordinates: GridCoordinates | None = None
    status: Literal["in_transit", "secured", "holding", "threat", "evac_requested", "unknown"] = "unknown"
    confidence: float = 0.5
    timestamp: str = ""
    raw_transcript: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.coordinates:
            data["coordinates"] = self.coordinates.to_dict()
        return data

    def validate(self) -> bool:
        """Validate safety rules against fabricated data."""
        if not self.entity or self.entity == "unknown":
            self.event_type = "unknown"
            self.confidence = min(self.confidence, 0.4)
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"Confidence out of range: {self.confidence}")
        return True
