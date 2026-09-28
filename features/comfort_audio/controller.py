"""Ducking and envelope controller for Adaptive Auditory Comfort Layer."""
from __future__ import annotations

import math
import numpy as np
from .config import ComfortAudioConfig


class ComfortAudioController:
    """Calculates smooth dynamic gain envelopes based on speech energy and transient events."""

    def __init__(self, config: ComfortAudioConfig | None = None):
        self.cfg = config or ComfortAudioConfig()
        self.cfg.validate()
        self.current_gain: float = 1.0 if self.cfg.enabled else 0.0
        self.hold_samples_remaining: int = 0

    def compute_energy_db(self, audio: np.ndarray) -> float:
        """Calculate RMS level in dBFS."""
        if len(audio) == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
        return 20.0 * math.log10(max(rms, 1e-6))

    def process_frame(
        self,
        speech_chunk: np.ndarray,
        sample_rate: int = 16000,
        has_transient: bool = False,
    ) -> np.ndarray:
        """Compute sample-level gain envelope for speech_chunk.
        
        Returns:
            np.ndarray: Array of shape (len(speech_chunk),) with gains in [0.0, 1.0].
        """
        n_samples = len(speech_chunk)
        if n_samples == 0:
            return np.empty(0, dtype=np.float32)

        if not self.cfg.enabled:
            return np.zeros(n_samples, dtype=np.float32)

        # If communication priority is disabled, maintain constant volume
        if not self.cfg.communication_priority:
            return np.full(n_samples, self.cfg.volume, dtype=np.float32)

        # Immediate mute if a blast / gunshot transient is flagged
        if self.cfg.transient_ducking and has_transient:
            self.current_gain = 0.0
            self.hold_samples_remaining = int(self.cfg.hold_ms * sample_rate / 1000.0)
            return np.zeros(n_samples, dtype=np.float32)

        # Check energy in this chunk
        energy_db = self.compute_energy_db(speech_chunk)
        speech_active = energy_db > self.cfg.speech_threshold_db

        # Rate of gain change per sample
        fade_out_samples = max(1, int(self.cfg.fade_out_ms * sample_rate / 1000.0))
        fade_in_samples = max(1, int(self.cfg.fade_in_ms * sample_rate / 1000.0))
        hold_samples = int(self.cfg.hold_ms * sample_rate / 1000.0)

        step_down = 1.0 / fade_out_samples
        step_up = 1.0 / fade_in_samples

        envelope = np.empty(n_samples, dtype=np.float32)

        for i in range(n_samples):
            if speech_active:
                # Fast duck down to 0
                self.current_gain = max(0.0, self.current_gain - step_down)
                self.hold_samples_remaining = hold_samples
            else:
                if self.hold_samples_remaining > 0:
                    self.hold_samples_remaining -= 1
                else:
                    # Gradual fade back in to 1.0
                    self.current_gain = min(1.0, self.current_gain + step_up)

            envelope[i] = self.current_gain * self.cfg.volume

        return envelope

    def reset(self) -> None:
        """Reset state."""
        self.current_gain = 1.0 if self.cfg.enabled else 0.0
        self.hold_samples_remaining = 0
