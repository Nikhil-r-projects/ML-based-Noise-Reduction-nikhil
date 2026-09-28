# Real-Time Situational Awareness Map (`features/situational_awareness`)

## Purpose
Extracts structured tactical battlefield events from transcribed radio communications and updates a real-time tactical map.

## Components
- `schema.py`: `TacticalEvent` and `GridCoordinates` with safety validation (zero hallucination).
- `parser.py`: `RadioProtocolParser` mapping callsigns (`Alpha`, `Bravo`), verbs (`moving`, `reached`, `contact`), and waypoints.
- `event_bus.py`: `EventBus` managing subscriptions and history for real-time web push (SSE / WebSocket).

## Quick Example
```python
from features.situational_awareness import RadioProtocolParser, GLOBAL_EVENT_BUS

parser = RadioProtocolParser()
event = parser.parse("Alpha team moving towards checkpoint Bravo.")
GLOBAL_EVENT_BUS.publish(event)

print(event.to_dict())
```
