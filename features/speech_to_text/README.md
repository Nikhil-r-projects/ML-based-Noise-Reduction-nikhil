# Speech-to-Text Communication Backup (`features/speech_to_text`)

## Purpose
Provides an asynchronous, non-blocking speech recognition pipeline converting enhanced audio into timestamped tactical transcripts.

## Key Features
- **Strictly Asynchronous**: Operates via a bounded background queue (`ASRWorkerThread`); never adds latency to the hard real-time ANC path.
- **Pluggable Backends**: Supports local quantized `faster-whisper` and offline deterministic test backends.
- **Native 16 kHz Match**: Consumes 16,000 Hz float32 audio directly from `ANCNetV3` without resampling overhead.

## Quick Example
```python
from features.speech_to_text import get_asr_engine, ASRConfig

engine = get_asr_engine(ASRConfig(backend="mock"))
result = engine.transcribe(enhanced_audio, sample_rate=16000)
print(f"[{result.timestamp}] {result.text}")
```
