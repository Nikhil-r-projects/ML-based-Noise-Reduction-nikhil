"""Audio bed synthesizer and mixer for Adaptive Auditory Comfort Layer."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import soundfile as sf
from .config import ComfortAudioConfig
from .controller import ComfortAudioController


class ComfortAudioMixer:
    """Synthesizes neutral acoustic beds and blends them with the enhanced speech stream."""

    def __init__(self, config: ComfortAudioConfig | None = None):
        self.cfg = config or ComfortAudioConfig()
        self.cfg.validate()
        self.controller = ComfortAudioController(self.cfg)
        self.phase: float = 0.0
        # Pink noise filter state (Kellet's filter)
        self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.ambient_cache: np.ndarray | None = None
        self.ambient_idx: int = 0
        if self.cfg.ambient_file_path and Path(self.cfg.ambient_file_path).exists():
            data, fs = sf.read(self.cfg.ambient_file_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            self.ambient_cache = data

    def generate_bed(self, n_samples: int, sample_rate: int = 16000) -> np.ndarray:
        """Generate n_samples of the chosen comfort audio bed normalized to [-1, 1]."""
        if n_samples == 0:
            return np.empty(0, dtype=np.float32)

        if self.cfg.mode == "neutral_tone":
            # Low-frequency soft sine hum with gentle 2nd harmonic
            t = np.arange(n_samples, dtype=np.float32) / sample_rate
            f0 = self.cfg.tone_freq_hz
            phases = 2.0 * math.pi * f0 * t + self.phase
            self.phase = float((phases[-1] + 2.0 * math.pi * f0 / sample_rate) % (2.0 * math.pi))
            wave = 0.85 * np.sin(phases) + 0.15 * np.sin(2.0 * phases)
            return wave.astype(np.float32)

        elif self.cfg.mode == "ambient_file" and self.ambient_cache is not None:
            # Loop cached ambient file
            buf = np.empty(n_samples, dtype=np.float32)
            cache_len = len(self.ambient_cache)
            for i in range(n_samples):
                buf[i] = self.ambient_cache[self.ambient_idx]
                self.ambient_idx = (self.ambient_idx + 1) % cache_len
            return buf

        else:
            # Default: Pink noise via Paul Kellet's filter approximation
            white = np.random.uniform(-1.0, 1.0, size=n_samples).astype(np.float32)
            pink = np.empty(n_samples, dtype=np.float32)
            b0, b1, b2, b3, b4, b5, b6 = self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6
            for i in range(n_samples):
                w = white[i]
                b0 = 0.99886 * b0 + w * 0.0555179
                b1 = 0.99332 * b1 + w * 0.0750759
                b2 = 0.96900 * b2 + w * 0.1538520
                b3 = 0.86650 * b3 + w * 0.3104856
                b4 = 0.55000 * b4 + w * 0.5329522
                b5 = -0.7616 * b5 - w * 0.0168980
                pink[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11
                b6 = w * 0.115926
            self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6 = b0, b1, b2, b3, b4, b5, b6
            return np.clip(pink, -1.0, 1.0).astype(np.float32)

    def mix(
        self,
        speech: np.ndarray,
        sample_rate: int = 16000,
        has_transient: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mix speech and adaptive comfort bed.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: (mixed_audio, gain_envelope)
        """
        n_samples = len(speech)
        if not self.cfg.enabled or n_samples == 0:
            return speech.copy(), np.zeros(n_samples, dtype=np.float32)

        envelope = self.controller.process_frame(speech, sample_rate, has_transient)
        bed = self.generate_bed(n_samples, sample_rate)
        mixed = speech + bed * envelope
        return np.clip(mixed, -1.0, 1.0).astype(np.float32), envelope

    def reset(self) -> None:
        self.controller.reset()
        self.phase = 0.0
        self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.ambient_idx = 0
