"""HTTP API for ZeroTTS.

    pip install "zerotts[api]"   # or: pip install fastapi uvicorn
    pip install -e .            # from this repo
    python api/app.py
    python api/app.py --host 0.0.0.0 --port 8000 --model ./local_dir

Endpoints:
    GET  /health
    GET  /voices
    POST /v1/tts          → audio/wav
    POST /v1/tts/base64   → JSON {audio_base64, sample_rate, duration_sec}
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
if os.path.isfile(os.path.join(_SRC, "zerotts", "__init__.py")):
    sys.path.insert(0, _SRC)

from zerotts import ZeroTTS, normalize_vi_text  # noqa: E402
from zerotts.audio import concat_with_silence  # noqa: E402
from zerotts.chunking import (  # noqa: E402
    chunk_text,
    clean_segment_punctuation,
    normalize_punctuation,
)

# Set by main() / lifespan before serving requests.
_tts: ZeroTTS | None = None
_settings: dict = {}


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize.")
    voice: Optional[str] = Field(
        "maichi", description="Voice pack name. null = unconditional.")
    cfg_scale: float = 1.0
    audio_temperature: float = 0.8
    audio_topk: int = 25
    audio_topp: float = 0.95
    audio_repetition_penalty: float = 1.2
    text_norm: bool = Field(
        True, description="Apply Vietnamese date/number/acronym expansion.")
    chunk: bool = Field(
        False, description="Split long text into segments before synthesis.")
    max_chunk_sec: float = 15.0
    gap_sec: float = 0.15
    seed: Optional[int] = None


class VoiceInfo(BaseModel):
    name: str
    display_name: Optional[str] = None
    gender: Optional[str] = None
    language: Optional[str] = None
    tags: list[str] = []
    description: Optional[str] = None


class TTSBase64Response(BaseModel):
    audio_base64: str
    format: str = "wav"
    sample_rate: int
    duration_sec: float
    elapsed_sec: float
    voice: Optional[str] = None


def _engine() -> ZeroTTS:
    if _tts is None:
        raise HTTPException(503, "Model not loaded yet.")
    return _tts


def _synthesize(req: TTSRequest) -> tuple[np.ndarray, float]:
    tts = _engine()
    if req.seed is not None:
        np.random.seed(req.seed)

    text = req.text
    if req.text_norm:
        text = normalize_vi_text(text)

    segments = [text]
    if req.chunk:
        segments = [
            clean_segment_punctuation(s)
            for s in chunk_text(normalize_punctuation(text),
                                max_chunk_sec=req.max_chunk_sec)
        ]
        segments = [s for s in segments if s]
        if not segments:
            raise HTTPException(400, "Text is empty after chunking.")

    kwargs = {
        "voice": req.voice,
        "cfg_scale": req.cfg_scale,
        "audio_temperature": req.audio_temperature,
        "audio_topk": req.audio_topk,
        "audio_topp": req.audio_topp,
        "audio_repetition_penalty": req.audio_repetition_penalty,
    }

    t0 = time.perf_counter()
    chunks = [tts.synthesize(seg, **kwargs) for seg in segments]
    audio = concat_with_silence(chunks, req.gap_sec, tts.sample_rate)
    elapsed = time.perf_counter() - t0
    return audio, elapsed


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio).squeeze(), sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _tts
        model = _settings.get("model", "zeroweight-ai/ZeroTTS")
        threads = int(_settings.get("threads", 4))
        print(f"Loading model: {model} (threads={threads}) …")
        _tts = ZeroTTS.from_pretrained(model, intra_op_num_threads=threads)
        for path in _settings.get("voices", []):
            names = _tts.add_voices(path)
            print(f"Added voices from {path}: {names}")
        print(f"Ready. Voices: {_tts.list_voices()}")
        yield
        _tts = None

    app = FastAPI(
        title="ZeroTTS API",
        description="Vietnamese zero-shot text-to-speech HTTP API.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        ready = _tts is not None
        return {
            "status": "ok" if ready else "loading",
            "model": _settings.get("model"),
            "sample_rate": getattr(_tts, "sample_rate", None),
        }

    @app.get("/voices", response_model=list[VoiceInfo])
    def voices():
        tts = _engine()
        out: list[VoiceInfo] = []
        for name in tts.list_voices():
            try:
                v = tts.load_voice(name)
                out.append(VoiceInfo(
                    name=name,
                    display_name=getattr(v, "display_name", None),
                    gender=getattr(v, "gender", None),
                    language=getattr(v, "language", None),
                    tags=list(getattr(v, "tags", []) or []),
                    description=getattr(v, "description", None) or None,
                ))
            except Exception:
                out.append(VoiceInfo(name=name))
        return out

    @app.post("/v1/tts")
    def tts_wav(req: TTSRequest):
        """Synthesize text and return a WAV file."""
        try:
            audio, _elapsed = _synthesize(req)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, str(e)) from e

        tts = _engine()
        wav = _audio_to_wav_bytes(audio, tts.sample_rate)
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="speech.wav"',
                "X-Sample-Rate": str(tts.sample_rate),
                "X-Duration-Sec": f"{audio.shape[-1] / tts.sample_rate:.3f}",
            },
        )

    @app.post("/v1/tts/base64", response_model=TTSBase64Response)
    def tts_base64(req: TTSRequest):
        """Synthesize text and return base64-encoded WAV in JSON."""
        try:
            audio, elapsed = _synthesize(req)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, str(e)) from e

        tts = _engine()
        wav = _audio_to_wav_bytes(audio, tts.sample_rate)
        return TTSBase64Response(
            audio_base64=base64.b64encode(wav).decode("ascii"),
            sample_rate=tts.sample_rate,
            duration_sec=audio.shape[-1] / tts.sample_rate,
            elapsed_sec=elapsed,
            voice=req.voice,
        )

    return app


app = create_app()


def main() -> None:
    p = argparse.ArgumentParser(description="ZeroTTS HTTP API server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="zeroweight-ai/ZeroTTS",
                   help="HF repo id or local model directory.")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--voice", action="append", default=[], dest="voices",
                   help="Extra voice zip/dir (repeatable).")
    args = p.parse_args()

    _settings.update({
        "model": args.model,
        "threads": args.threads,
        "voices": args.voices,
    })

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
