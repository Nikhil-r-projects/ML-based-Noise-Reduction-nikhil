"""Asynchronous non-blocking audio queue and ASR worker thread."""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable
import numpy as np
from .asr_engine import ASREngine, get_asr_engine
from .config import ASRConfig
from .models import TranscriptResult


class ASRWorkerThread:
    """Background worker thread that consumes audio chunks from a non-blocking queue."""

    def __init__(
        self,
        engine: ASREngine | None = None,
        config: ASRConfig | None = None,
        on_transcript: Callable[[TranscriptResult], None] | None = None,
    ):
        self.cfg = config or ASRConfig()
        self.engine = engine or get_asr_engine(self.cfg)
        self.on_transcript = on_transcript
        self.audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=self.cfg.queue_maxsize)
        self._running = False
        self._thread: threading.Thread | None = None
        self.latest_result: TranscriptResult | None = None

    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="ASRWorkerThread")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Gracefully stop the worker thread."""
        if not self._running:
            return
        self._running = False
        try:
            self.audio_queue.put_nowait(None)  # Sentinel to exit
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def submit_audio(self, audio: np.ndarray) -> bool:
        """Submit an audio chunk for asynchronous transcription.
        
        This method is strictly non-blocking. If the queue is full, the oldest
        chunk is dropped to prioritize the most recent transmission.
        
        Returns:
            bool: True if queued, False if dropped or worker not running.
        """
        if not self._running:
            return False

        try:
            self.audio_queue.put_nowait(audio.copy())
            return True
        except queue.Full:
            # Drop oldest chunk to prevent buffer buildup
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(audio.copy())
                return True
            except queue.Full:
                return False

    def _worker_loop(self) -> None:
        while self._running:
            try:
                item = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:  # Stop sentinel
                break

            try:
                result = self.engine.transcribe(item, sample_rate=self.cfg.sample_rate)
                self.latest_result = result
                if self.on_transcript and result.text:
                    self.on_transcript(result)
            except Exception as e:
                print(f"[ASRWorkerThread Error] {e}")
            finally:
                self.audio_queue.task_done()
