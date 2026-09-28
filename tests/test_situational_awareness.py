"""Tests for Situational Awareness and Radio Protocol Parser."""
import pytest
from features.situational_awareness import (
    RadioProtocolParser,
    TacticalEvent,
    GridCoordinates,
    EventBus,
)


def test_parser_movement_event():
    parser = RadioProtocolParser()
    text = "Alpha team moving towards checkpoint Bravo."
    event = parser.parse(text)

    assert event.entity == "Alpha"
    assert event.event_type == "movement"
    assert event.status == "in_transit"
    assert event.destination == "Checkpoint Bravo"
    assert event.coordinates is not None
    assert event.coordinates.x == 380
    assert event.coordinates.y == 180
    assert event.confidence >= 0.85
    assert event.validate() is True


def test_parser_arrival_event():
    parser = RadioProtocolParser()
    text = "Charlie team reached sector four."
    event = parser.parse(text)

    assert event.entity == "Charlie"
    assert event.event_type == "arrival"
    assert event.status == "secured"
    assert event.location == "Sector Four"
    assert event.coordinates is not None
    assert event.coordinates.x == 160
    assert event.coordinates.y == 360


def test_parser_contact_event():
    parser = RadioProtocolParser()
    text = "Delta squad under fire at objective Iron."
    event = parser.parse(text)

    assert event.entity == "Delta"
    assert event.event_type == "contact"
    assert event.status == "threat"
    assert event.location == "Objective Iron"
    assert event.coordinates is not None
    assert event.coordinates.x == 280
    assert event.coordinates.y == 240


def test_parser_ambiguous_text_no_hallucination():
    parser = RadioProtocolParser()
    text = "Static on the radio, repeat transmission."
    event = parser.parse(text)

    assert event.entity == "Unknown"
    assert event.event_type == "unknown"
    assert event.destination is None
    assert event.location is None
    assert event.coordinates is None
    assert event.confidence < 0.5


def test_event_bus_pub_sub():
    bus = EventBus(max_history=5)
    sub = bus.subscribe()

    event = TacticalEvent(
        entity="Bravo",
        event_type="holding",
        location="Checkpoint 2",
        status="holding",
        confidence=0.9,
    )
    bus.publish(event)

    history = bus.get_history()
    assert len(history) == 1
    assert history[0]["entity"] == "Bravo"

    received = sub.get_nowait()
    assert received.entity == "Bravo"
