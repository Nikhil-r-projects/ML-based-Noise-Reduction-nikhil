"""Deterministic military radio protocol parser and entity extractor."""
from __future__ import annotations

import datetime
import re
from .schema import GridCoordinates, TacticalEvent

# Calibrated synthetic grid coordinates for demo visualization
KNOWN_LOCATIONS: dict[str, tuple[int, int]] = {
    "checkpoint bravo": (380, 180),
    "checkpoint 2": (220, 290),
    "checkpoint two": (220, 290),
    "checkpoint alpha": (140, 120),
    "checkpoint 1": (140, 120),
    "sector four": (160, 360),
    "sector 4": (160, 360),
    "sector seven": (440, 320),
    "sector 7": (440, 320),
    "objective iron": (280, 240),
    "landing zone alpha": (400, 100),
    "base": (80, 420),
    "hq": (80, 420),
}

CALLSIGNS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
    "golf", "hotel", "viper", "ghost", "recon", "eagle", "falcon", "iron",
]


class RadioProtocolParser:
    """Extracts tactical entities and events from transcribed radio speech."""

    def __init__(self, locations: dict[str, tuple[int, int]] | None = None):
        self.locations = {k.lower(): v for k, v in (locations or KNOWN_LOCATIONS).items()}

    def parse(self, text: str, timestamp: str | None = None) -> TacticalEvent:
        if not timestamp:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        cleaned = text.strip()
        lower = cleaned.lower()

        # 1. Identify Callsign / Unit
        entity: str | None = None
        for cs in CALLSIGNS:
            pattern = rf"\b{cs}\b(\s+(team|squad|element|recon|lead))?"
            m = re.search(pattern, lower)
            if m:
                entity = cs.capitalize()
                break

        if not entity:
            return TacticalEvent(
                entity="Unknown",
                event_type="unknown",
                status="unknown",
                confidence=0.2,
                timestamp=timestamp,
                raw_transcript=cleaned,
            )

        # 2. Identify Event Type, Status & Locations
        event_type = "unknown"
        status = "unknown"
        destination: str | None = None
        location: str | None = None
        confidence = 0.85

        # Movement
        m_move = re.search(r"(?:moving (?:towards|to)|advancing (?:towards|to)|en route to|pushing to)\s+([^.,!]+)", lower)
        # Arrival
        m_arr = re.search(r"(?:reached|arrived at|secured)\s+([^.,!]+)", lower)
        # Holding / Waiting
        m_hold = re.search(r"(?:waiting at|holding (?:position at|at)|standing by at)\s+([^.,!]+)", lower)
        # Contact / Under fire
        m_contact = re.search(r"(?:under fire at|contact (?:with enemy|at)|engaging at|taking fire at)\s+([^.,!]+)", lower)
        # Evac
        m_evac = re.search(r"(?:requesting evac at|need extraction at|medevac (?:needed at|at))\s+([^.,!]+)", lower)

        if m_contact:
            event_type = "contact"
            status = "threat"
            location = self._clean_loc(m_contact.group(1))
            confidence = 0.95
        elif m_evac:
            event_type = "evacuation"
            status = "evac_requested"
            location = self._clean_loc(m_evac.group(1))
            confidence = 0.95
        elif m_move:
            event_type = "movement"
            status = "in_transit"
            destination = self._clean_loc(m_move.group(1))
            confidence = 0.90
        elif m_arr:
            event_type = "arrival"
            status = "secured"
            location = self._clean_loc(m_arr.group(1))
            confidence = 0.92
        elif m_hold:
            event_type = "holding"
            status = "holding"
            location = self._clean_loc(m_hold.group(1))
            confidence = 0.88

        # 3. Resolve Coordinates
        target_loc = destination or location
        coords: GridCoordinates | None = None
        if target_loc:
            coords = self._lookup_coords(target_loc)

        return TacticalEvent(
            entity=entity,
            event_type=event_type,  # type: ignore
            destination=destination,
            location=location,
            coordinates=coords,
            status=status,  # type: ignore
            confidence=round(confidence, 2),
            timestamp=timestamp,
            raw_transcript=cleaned,
        )

    def _clean_loc(self, raw: str) -> str:
        s = raw.strip()
        s = re.sub(r"\s+", " ", s)
        # Capitalize words
        return " ".join(w.capitalize() for w in s.split())

    def _lookup_coords(self, loc_name: str) -> GridCoordinates | None:
        key = loc_name.lower().strip()
        if key in self.locations:
            x, y = self.locations[key]
            return GridCoordinates(x=x, y=y)
        # Check partial match
        for known, (x, y) in self.locations.items():
            if known in key or key in known:
                return GridCoordinates(x=x, y=y)
        return None
