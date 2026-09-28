"""Thread-safe event bus and broadcast broker for situational awareness."""
from __future__ import annotations

import queue
import threading
from typing import Callable
from .schema import TacticalEvent


class EventBus:
    """Pub/Sub broker and event log for tactical updates."""

    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self._history: list[TacticalEvent] = []
        self._subscribers: list[queue.Queue[TacticalEvent]] = []
        self._lock = threading.Lock()

    def publish(self, event: TacticalEvent) -> None:
        """Publish an event to all active subscribers and append to history."""
        with self._lock:
            self._history.append(event)
            if len(self._history) > self.max_history:
                self._history.pop(0)

            # Broadcast to queues
            dead_subscribers = []
            for sub in self._subscribers:
                try:
                    sub.put_nowait(event)
                except queue.Full:
                    dead_subscribers.append(sub)

            for dead in dead_subscribers:
                self._subscribers.remove(dead)

    def subscribe(self, maxsize: int = 20) -> queue.Queue[TacticalEvent]:
        """Subscribe to real-time events. Returns a thread-safe Queue."""
        sub: queue.Queue[TacticalEvent] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(sub)
        return sub

    def unsubscribe(self, sub: queue.Queue[TacticalEvent]) -> None:
        with self._lock:
            if sub in self._subscribers:
                self._subscribers.remove(sub)

    def get_history(self) -> list[dict]:
        with self._lock:
            return [ev.to_dict() for ev in self._history]


# Singleton instance for system-wide event routing
GLOBAL_EVENT_BUS = EventBus()
