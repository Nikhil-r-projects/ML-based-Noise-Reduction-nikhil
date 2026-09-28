# Adaptive Auditory Comfort Layer (`features/comfort_audio`)

## Purpose
Provides an optional low-level acoustic layer (pink noise, harmonic tone, or ambient file) with dynamic sidechain ducking. The comfort layer automatically yields priority to incoming communications and acoustic transient events (gunfire, blast overpressure).

## Components
- `config.py`: `ComfortAudioConfig` dataclass specifying volume, mode, thresholds, and fade timing.
- `controller.py`: `ComfortAudioController` computing smooth exponential gain envelopes.
- `mixer.py`: `ComfortAudioMixer` synthesizing audio beds and summing them with the speech signal.

## Quick Example
```python
from features.comfort_audio import ComfortAudioConfig, ComfortAudioMixer

cfg = ComfortAudioConfig(enabled=True, volume=0.2, mode="pink_noise")
mixer = ComfortAudioMixer(cfg)
mixed_audio, gain_envelope = mixer.mix(enhanced_speech, sample_rate=16000)
```
