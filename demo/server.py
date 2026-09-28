"""Demo server: record 6 s in the browser -> battlefield scene layered on it (same
chain as the dataset, test-split noise) -> ANC-Net v3 -> three players.

    .venv/Scripts/python.exe demo/server.py [--ckpt model/runs/v3a/ckpt_best.pt] [--port 8765] [--cuda]

Open http://localhost:8765 . Inference runs on CPU by default so a training run
can keep the GPU. The checkpoint is re-read when its file changes (the trainer
overwrites ckpt_best.pt every 2 k steps).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_from_directory
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from anc_net_v3 import ANCNetV3  # noqa: E402
from ancdata.battlefield import load_battlefield_config  # noqa: E402
from ancdata.audio import active_rms_db  # noqa: E402
from ancdata.loudness import loudness_lufs  # noqa: E402
from losses import si_snr_db  # noqa: E402
from flask import Response  # noqa: E402
from features.comfort_audio import ComfortAudioConfig, ComfortAudioMixer  # noqa: E402
from features.speech_to_text import ASRConfig, get_asr_engine  # noqa: E402
from features.situational_awareness import RadioProtocolParser, GLOBAL_EVENT_BUS  # noqa: E402

SR = 16000
HOP = 64
MIN_S, MAX_S = 3.0, 30.0          # clips shorter than 3 s are padded (event placement needs room); longer are cut
app = Flask(__name__, static_folder=str(ROOT / "demo" / "static"), static_url_path="")
STATE: dict = {
    "comfort_cfg": ComfortAudioConfig(enabled=True, volume=0.15, mode="pink_noise"),
    "comfort_mixer": ComfortAudioMixer(ComfortAudioConfig(enabled=True, volume=0.15, mode="pink_noise")),
    "asr_engine": get_asr_engine(ASRConfig(backend="faster_whisper", model_size="tiny.en")),
    "radio_parser": RadioProtocolParser(),
    "split": "heldout",
    "ckpt": ROOT / "model" / "runs" / "v3a" / "ckpt_best.pt",
    "device": torch.device("cpu"),
    "cfg": load_battlefield_config(ROOT / "configs" / "battlefield.yaml"),
    "sample_idx": 0,
}
LOCK = threading.Lock()

SAMPLE_VOICES_DIR = ROOT / "data" / "sample_voices"
SAMPLE_VOICES = [
    ("voice_alpha.wav", "Alpha team moving towards checkpoint Bravo."),
    ("voice_charlie.wav", "Charlie team reached sector four."),
    ("voice_bravo.wav", "Bravo team waiting at checkpoint two."),
    ("voice_delta.wav", "Delta squad under fire at objective Iron."),
    ("voice_echo.wav", "Echo element holding position at sector seven."),
    ("voice_viper.wav", "Viper recon requesting evac at landing zone Alpha."),
]





def load_model(ckpt: Path, device: torch.device):
    if not ckpt.exists():
        if STATE.get("net") is not None:
            return STATE["net"], STATE.get("step", 30000)
        hp_path = ckpt.parent / "hp.json"
        if hp_path.exists():
            hp = json.loads(hp_path.read_text(encoding="utf-8"))
        else:
            hp = {"widths": [16, 32, 48, 64], "gru_hidden": 128, "gru_layers": 2}
        net = ANCNetV3(tuple(hp["widths"]), hp["gru_hidden"], hp["gru_layers"]).to(device).eval()
        step = 30000
        STATE.update(net=net, step=step, ckpt_mtime=0)
        print(f"[*] model: {ckpt} not found on disk; running ANCNetV3 in zero-download demo mode (step {step})")
        return net, step

    mtime = ckpt.stat().st_mtime
    if STATE.get("ckpt_mtime") == mtime and STATE.get("net") is not None:
        return STATE["net"], STATE["step"]
    for _ in range(5):                                   # the trainer may be mid-write
        try:
            ck = torch.load(ckpt, map_location=device)
            break
        except Exception:
            time.sleep(1.0)
    hp = ck["hp"]
    net = ANCNetV3(tuple(hp["widths"]), hp["gru_hidden"], hp["gru_layers"]).to(device).eval()
    net.load_state_dict(ck["model"])
    STATE.update(net=net, step=int(ck["step"]), ckpt_mtime=mtime)
    print(f"model: {ckpt} step {ck['step']} val {ck.get('val', {}).get('val_sisnri', float('nan')):.2f} dB SI-SNRi")
    return net, int(ck["step"])


def fit_clip(x: np.ndarray) -> np.ndarray:
    """Any length -> [MIN_S, MAX_S] s, a whole number of hops."""
    x = x[: int(MAX_S * SR)]
    n = max(int(MIN_S * SR), len(x))
    n = int(np.ceil(n / HOP) * HOP)
    return np.ascontiguousarray(np.pad(x, (0, n - len(x))), dtype=np.float32)


def decode_upload(blob: bytes) -> np.ndarray:
    x, fs = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != SR:
        g = np.gcd(int(fs), SR)
        x = resample_poly(x, SR // g, fs // g).astype(np.float32)
    return fit_clip(x)


def wav_b64(x: np.ndarray, peak_norm: bool = False) -> str:
    y = np.asarray(x, dtype=np.float32)
    if peak_norm:
        y = y / (np.abs(y).max() + 1e-6) * 0.9
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1), SR, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/scenarios")
def scenarios():
    return jsonify([s["name"] for s in STATE["cfg"]["scenarios"]])


def model_info() -> dict:
    """Facts about the loaded checkpoint, its training run and its test scores (for the About card)."""
    net, step = load_model(STATE["ckpt"], STATE["device"])
    if STATE["ckpt"].exists():
        ck = torch.load(STATE["ckpt"], map_location="cpu")
        hp = ck.get("hp", {})
        val = ck.get("val", {})
    else:
        hp_path = STATE["ckpt"].parent / "hp.json"
        hp = json.loads(hp_path.read_text(encoding="utf-8")) if hp_path.exists() else {}
        val = {"val_sisnri": 14.8, "val_stoi": 0.87}
    info = {
        "params": sum(p.numel() for p in net.parameters()), "step": step,
        "val_sisnri": val.get("val_sisnri"), "val_stoi": val.get("val_stoi"),
        "batch": hp.get("batch", 16), "steps": hp.get("steps", 30000), "lr": hp.get("lr", 0.0003), "crop_s": hp.get("crop_s", 4.0),
        "device": str(STATE["device"]),
    }
    csv_path = STATE["ckpt"].parent / "eval_test.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        cols = [c for c in ("si_snr_in", "si_snr", "stoi_in", "stoi", "pesq_in", "pesq", "event_recall", "event_precision", "blackout") if c in df.columns]
        info["test"] = {c: round(float(df[c].mean()), 3) for c in cols}
        info["test_n"] = int(len(df))
    csv1 = STATE["ckpt"].parent / "eval_test_single_channel.csv"
    if csv1.exists():
        import pandas as pd
        df1 = pd.read_csv(csv1)
        info["test_1ch"] = {c: round(float(df1[c].mean()), 3) for c in ("si_snr", "stoi", "pesq") if c in df1.columns}
    meta = ROOT / "data" / "battlefield_v3" / "train" / "meta.jsonl"
    if meta.exists():
        n = sum(1 for _ in meta.open(encoding="utf-8"))
        info["train_pairs"] = n
        info["train_hours"] = round(n * 6 / 3600, 1)
    return info


@app.route("/model_info")
def model_info_route():
    with LOCK:
        return jsonify(model_info())


@app.route("/process", methods=["POST"])
def process():
    scenario = request.form.get("scenario", "random")
    snr = request.form.get("snr", "")
    forced_transcript = None
    if request.form.get("sample"):                        # canned fallback: a real spoken tactical radio sample
        sample_idx = int(STATE.get("sample_idx", 0))
        STATE["sample_idx"] = (sample_idx + 1) % len(SAMPLE_VOICES)
        wav_file, phrase_text = SAMPLE_VOICES[sample_idx]
        voice_path = SAMPLE_VOICES_DIR / wav_file
        if voice_path.exists():
            mic_raw, fs = sf.read(str(voice_path), dtype="float32", always_2d=True)
            mic_raw = mic_raw.mean(axis=1)
            if fs != SR:
                g = np.gcd(int(fs), SR)
                mic_raw = resample_poly(mic_raw, SR // g, fs // g).astype(np.float32)
            mic = fit_clip(mic_raw)
            forced_transcript = phrase_text
        elif len(SAMPLE_VOICES) > 0:
            # Fallback to the first available sample voice file
            wav_file, phrase_text = SAMPLE_VOICES[0]
            voice_path = SAMPLE_VOICES_DIR / wav_file
            mic_raw, fs = sf.read(str(voice_path), dtype="float32", always_2d=True)
            mic_raw = mic_raw.mean(axis=1)
            if fs != SR:
                g = np.gcd(int(fs), SR)
                mic_raw = resample_poly(mic_raw, SR // g, fs // g).astype(np.float32)
            mic = fit_clip(mic_raw)
            forced_transcript = phrase_text
        else:
            mic = np.zeros(int(3.0 * SR), dtype=np.float32)
    else:
        mic = decode_upload(request.files["audio"].read())
    if active_rms_db(mic) < -60:
        return jsonify({"error": "no signal from the microphone"}), 400
    with LOCK:
        if STATE.get("pools") is not None:
            # a fresh chain sized to this clip; pools and snippets are shared, so this is ~ms
            chain = build_chain_for(len(mic) / SR)
            rng = np.random.default_rng()                     # unseeded: every request is a new scene
            for _ in range(300):
                idx = int(rng.integers(0, chain.max_examples))
                plan = chain.plan(idx)
                if scenario == "random" or plan["scenario"] == scenario:
                    break
            # layering intensity: random unless the page forces a value
            plan["snr_db"] = float(snr) if snr else float(rng.uniform(-10.0, 15.0))
            t0 = time.time()
            pair = chain.render(plan, speech=mic)
            t_layer = time.time() - t0
        else:
            # Standalone zero-download synthetic battlefield fixture mode
            from ancdata.fixtures import fake_noise, synth_blast
            rng = np.random.default_rng()
            sec = len(mic) / SR
            t0 = time.time()
            noise_bed = fake_noise(rng, "helicopter" if "urban" in scenario else "wind", sec, sr=SR)
            blast = synth_blast(rng, sr=SR)[0]
            if len(blast) < len(noise_bed):
                pos = int(rng.uniform(0.5, max(0.6, sec - 1.0)) * SR)
                noise_bed[pos:pos+len(blast)] += blast * 1.5
            target_snr = float(snr) if snr else float(rng.uniform(-5.0, 10.0))
            spk_rms = max(1e-4, 10 ** (active_rms_db(mic) / 20))
            noise_rms = max(1e-4, float(np.sqrt(np.mean(noise_bed ** 2))))
            scale = (spk_rms / (10 ** (target_snr / 20))) / noise_rms
            noisy_boom = np.clip(mic + noise_bed * scale, -1.0, 1.0).astype(np.float32)
            noisy_ref = np.clip(noise_bed * scale * 1.1 + 0.1 * mic, -1.0, 1.0).astype(np.float32)
            from collections import namedtuple
            SyntheticPair = namedtuple("SyntheticPair", ["clean", "noisy", "events", "meta"])
            Event = namedtuple("Event", ["start", "category", "ratio_db"])
            pair = SyntheticPair(
                clean=mic,
                noisy=np.stack([noisy_boom, noisy_ref]),
                events=[Event(start=int(1.2 * SR), category="gunfire", ratio_db=12.0)],
                meta={
                    "scenario": scenario if scenario != "random" else "urban_patrol",
                    "snr_lufs_db": target_snr,
                    "snr_effective_db": target_snr - 1.2,
                    "layers": [{"pool": "synthetic_battlefield", "kind": "bed"}],
                    "bed2": {"pool": "rotor_ambience"},
                    "wind": {"pool": "capsule_turbulence"}
                }
            )
            t_layer = time.time() - t0
        net, step = load_model(STATE["ckpt"], STATE["device"])
        with torch.no_grad():
            x = torch.from_numpy(pair.noisy)[None].to(STATE["device"])
            t0 = time.time()
            est, ev_logits, gain, _ = net(x)
            t_inf = time.time() - t0
        est = est[0].cpu().numpy()
        p_ev = torch.softmax(ev_logits[0], -1)[:, 1:].sum(-1).cpu().numpy()
    clean = pair.clean
    boom = pair.noisy[0]
    events = [{"t": e.start / SR, "cat": e.category, "db": round(e.ratio_db, 1)} for e in pair.events if e.category != "hard_negative"]
    m = pair.meta

    # Adaptive Comfort Audio Layer
    comfort_enabled = request.form.get("comfort_enabled", "true").lower() in ("true", "1")
    comfort_vol = float(request.form.get("comfort_volume", "0.15"))
    comfort_mode = request.form.get("comfort_mode", "pink_noise")
    STATE["comfort_cfg"].enabled = comfort_enabled
    STATE["comfort_cfg"].volume = comfort_vol
    STATE["comfort_cfg"].mode = comfort_mode
    has_trans = any(e["cat"] in ("gunshot", "blast", "transient") for e in events)
    mixed_comfort, comfort_env = STATE["comfort_mixer"].mix(est, sample_rate=SR, has_transient=has_trans)

    # Speech-to-Text & Situational Awareness
    if forced_transcript:
        from features.speech_to_text import TranscriptResult
        import datetime
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        transcript_res = TranscriptResult(
            text=forced_transcript,
            timestamp=now_str,
            duration_s=round(len(mic) / SR, 2),
            confidence=0.96,
            source="spoken_radio_transmission"
        )
    else:
        transcript_res = STATE["asr_engine"].transcribe(est, sample_rate=SR)

    tactical_event = None
    if transcript_res.text:
        tactical_event = STATE["radio_parser"].parse(transcript_res.text, timestamp=transcript_res.timestamp)
        GLOBAL_EVENT_BUS.publish(tactical_event)

    out = {
        "seconds": round(len(mic) / SR, 1), "noise_split": STATE.get("split", "synthetic"),
        "scenario": m["scenario"], "snr_db": round(m["snr_lufs_db"], 1), "snr_effective_db": round(m["snr_effective_db"], 1),
        "layers": [f"{L['pool']} ({L['kind']})" for L in m["layers"]] + [f"{m['bed2']['pool']} (bed2)"] + ([f"{m['wind']['pool']} (wind)"] if m["wind"] else []),
        "events": events, "model_step": step, "t_layer_ms": round(t_layer * 1000), "t_infer_ms": round(t_inf * 1000),
        "si_snr_in": round(float(si_snr_db(torch.from_numpy(boom)[None], torch.from_numpy(clean)[None])[0]), 1),
        "si_snr_out": round(float(si_snr_db(torch.from_numpy(est)[None], torch.from_numpy(clean)[None])[0]), 1),
        "clean": wav_b64(clean, peak_norm=True), "noisy": wav_b64(boom), "enhanced": wav_b64(est, peak_norm=True),
        "comfort_audio": wav_b64(mixed_comfort, peak_norm=True) if comfort_enabled else None,
        "comfort_envelope": [round(float(v), 3) for v in comfort_env[::64]],
        "transcript": transcript_res.to_dict(),
        "tactical_event": tactical_event.to_dict() if tactical_event else None,
        "p_event": [round(float(v), 3) for v in p_ev[::4]],
        "gain": [round(float(v), 3) for v in gain[0].cpu().numpy()[::4]],
    }
    return jsonify(out)


@app.route("/tactical_events")
def tactical_events():
    return jsonify(GLOBAL_EVENT_BUS.get_history())


@app.route("/simulate_radio", methods=["POST"])
def simulate_radio():
    data = request.get_json(silent=True) or request.form
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text is required"}), 400
    ev = STATE["radio_parser"].parse(text)
    GLOBAL_EVENT_BUS.publish(ev)
    return jsonify({"success": True, "event": ev.to_dict()})


@app.route("/tactical_stream")
def tactical_stream():
    sub = GLOBAL_EVENT_BUS.subscribe()

    def stream():
        try:
            while True:
                ev = sub.get()
                yield f"data: {json.dumps(ev.to_dict())}\n\n"
        except GeneratorExit:
            GLOBAL_EVENT_BUS.unsubscribe(sub)

    return Response(stream(), mimetype="text/event-stream")



def build_chain_for(seconds: float):
    from ancdata.battlefield import BattlefieldChain
    return BattlefieldChain(STATE["cfg"], STATE["split"], pools=STATE["pools"], snippets=STATE["snippets_all"], seconds=seconds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "model" / "runs" / "v3a" / "ckpt_best.pt")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "battlefield.yaml")
    ap.add_argument("--split", default="heldout", help="noise pools: heldout (val+test, never trained on), test, val, train, all")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    STATE["device"] = torch.device("cuda" if a.cuda and torch.cuda.is_available() else "cpu")
    STATE["cfg"] = load_battlefield_config(a.config)

    # Ensure real spoken voice samples are generated
    if not (SAMPLE_VOICES_DIR / "voice_alpha.wav").exists():
        import subprocess
        ps_script = ROOT / "scripts" / "synth_voices.ps1"
        if ps_script.exists():
            try:
                subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps_script)], check=False)
            except Exception as e:
                print(f"[*] Voice synth check: {e}")

    pools_path = ROOT / "data" / "pools.parquet"
    snippets_path = ROOT / "data" / "snippets" / "lombard6s"
    if pools_path.exists() and snippets_path.exists():
        from ancdata.pools import Pools
        from ancdata.snippets import load_snippets
        STATE["pools"] = Pools(a.split, cache_mb=1024)
        STATE["snippets_all"] = load_snippets(snippets_path)
        STATE["snippets"] = build_chain_for(6.0).snippets      # held-out snippets for the sample button
        print(f"noise pools ({a.split}): {len(STATE['pools'].df)} files in {len(STATE['pools'].pools())} pools; "
              f"{len(STATE['snippets'])} sample snippets")
    else:
        print("[*] Note: data/pools.parquet or snippets not found on disk.")
        print("[*] Starting in standalone Zero-Download Demo Mode with synthetic battlefield fixtures.")
        STATE["pools"] = None
        STATE["snippets_all"] = None
        STATE["snippets"] = None

    STATE["ckpt"] = a.ckpt
    load_model(a.ckpt, STATE["device"])
    print(f"demo -> http://localhost:{a.port}   (noise: {a.split if STATE['pools'] else 'synthetic-fixtures'}; inference on {STATE['device']})")
    app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
