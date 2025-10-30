"""
Offline transcription with Whisper large-v3 (Transformers) using diarization segments produced earlier.

Inputs:
- Audio file (e.g., meeting.wav) 
- segments.json from diarization (list of {start, end, speaker})

Outputs (written to --outdir, default ./out):
- meeting_transcript.json : [{start, end, speaker, text, words:[{word,start,end}]}]
- meeting_transcript.md

Requirements:
  pip install torch transformers soundfile librosa tqdm
  whisper model downloaded
"""

from __future__ import annotations
import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline as hf_pipeline,
)

@dataclass
class Utterance:
    start: float
    end: float
    speaker: str
    text: str
    words: List[Dict[str, Any]]

def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def load_audio_16k_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return y.astype(np.float32, copy=False), sr
    except Exception:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype(np.float32, copy=False), target_sr



def load_segments(outdir: str) -> List[Dict[str, Any]]:
    seg_path = os.path.join(outdir, "segments.json")
    if not os.path.exists(seg_path):
        return [{"start": 0.0, "end": None, "speaker": "Speaker_0"}]
    with open(seg_path, "r", encoding="utf-8") as f:
        return json.load(f)

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline
import torch

def build_asr(
    asr_model: str,            
    device_choice: str = "auto",
    language: str = "auto",
    task: str = "transcribe",
    chunk_length_s: int | None = None,
    stride_left_s: int | None = None,
    stride_right_s: int | None = None,
):
    if device_choice == "auto":
        device = 0 if torch.cuda.is_available() else -1
    elif device_choice == "cuda":
        device = 0
    else:
        device = -1

    torch_dtype = torch.float16 if device == 0 else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_model,
        torch_dtype=torch_dtype,  
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,     
    )
    proc = AutoProcessor.from_pretrained(
        asr_model,
        local_files_only=True,    
    )

    generate_kwargs = {}
    if language and language != "auto":
        generate_kwargs["language"] = language
    if task:
        generate_kwargs["task"] = task

    kwargs = dict(
        model=model,
        tokenizer=proc.tokenizer,
        feature_extractor=proc.feature_extractor,
        device=device,                 
        return_timestamps="word",
        generate_kwargs=generate_kwargs,
    )
    if chunk_length_s:
        kwargs["chunk_length_s"] = int(chunk_length_s)
    if (stride_left_s or stride_right_s) is not None:
        kwargs["stride_length_s"] = (int(stride_left_s or 0), int(stride_right_s or 0))

    return hf_pipeline("automatic-speech-recognition", **kwargs)

def transcribe_segments(
    audio_path: str,
    outdir: str,
    asr_model: str,          
    device_choice: str,
    language: str = "auto",
    task: str = "transcribe",
    chunk_length_s: Optional[int] = None,
    stride_left_s: Optional[int] = None,
    stride_right_s: Optional[int] = None,
):
    os.makedirs(outdir, exist_ok=True)
    y, sr = load_audio_16k_mono(audio_path, target_sr=16000)
    segs = load_segments(outdir)

    asr = build_asr(
        asr_model=asr_model,
        device_choice=device_choice,
        language=language,
        task=task,
        chunk_length_s=chunk_length_s,
        stride_left_s=stride_left_s,
        stride_right_s=stride_right_s,
    )

    utterances: List[Utterance] = []
    for seg in tqdm(segs, desc="ASR by segment"):
        s0 = float(seg.get("start", 0.0))
        s1 = seg.get("end", None)
        if s1 is None or s1 <= 0:
            s1 = len(y) / sr

        a0, a1 = int(max(0.0, s0) * sr), int(max(0.0, s1) * sr)
        if a1 <= a0 + int(0.2 * sr):
            continue

        chunk = y[a0:a1]
        result = asr({"array": chunk, "sampling_rate": sr})

        text = (result.get("text") or "").strip()
        words: List[Dict[str, Any]] = []
        for w in result.get("chunks", []):
            ts = w.get("timestamp", [None, None])
            if not isinstance(ts, (list, tuple)) or ts[0] is None or ts[1] is None:
                continue
            words.append({
                "word": (w.get("text") or "").strip(),
                "start": s0 + float(ts[0]),
                "end":   s0 + float(ts[1]),
            })

        utterances.append(Utterance(
            start=s0,
            end=s1,
            speaker=str(seg.get("speaker", "Speaker")),
            text=text,
            words=words,
        ))

    merged: List[Utterance] = []
    for u in utterances:
        if merged and merged[-1].speaker == u.speaker and u.start - merged[-1].end <= 0.5:
            merged[-1].end = u.end
            merged[-1].text = (merged[-1].text + " " + u.text).strip()
            merged[-1].words.extend(u.words)
        else:
            merged.append(u)

    jpath = os.path.join(outdir, "meeting_transcript.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump([asdict(u) for u in merged], f, ensure_ascii=False, indent=2)

    mdpath = os.path.join(outdir, "meeting_transcript.md")
    with open(mdpath, "w", encoding="utf-8") as f:
        f.write("# Speaker-labeled Transcript (Whisper large-v3, offline)\n\n")
        for u in merged:
            f.write(f"**[{_ts(u.start)}–{_ts(u.end)}] {u.speaker}:** {u.text}\n\n")

    return jpath, mdpath, merged


# CLI
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Offline transcription of diarized segments with Whisper large-v3")
    ap.add_argument("--audio", required=True, help="Path to input audio (WAV recommended, 16 kHz mono)")
    ap.add_argument("--outdir", default="out", help="Directory containing segments.json; outputs written here")
    ap.add_argument(
        "--asr_model",
        required=True,
        help="LOCAL FOLDER of Whisper model (e.g., /home/jovyan/Oracle_local/models/whisper-large-v3)",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Where to run the model")
    ap.add_argument("--language", default="auto", help='ISO code like "en" or "auto" to detect')
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"],
                    help="Transcribe (same language) or translate to English")
    ap.add_argument("--chunk_length_s", type=int, default=None, help="Chunk length for long segments (e.g., 30)")
    ap.add_argument("--stride_left_s", type=int, default=None, help="Left stride seconds (e.g., 5)")
    ap.add_argument("--stride_right_s", type=int, default=None, help="Right stride seconds (e.g., 2)")
    return ap.parse_args()


def main() -> None:
    a = _parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    jpath, mdpath, merged = transcribe_segments(
        audio_path=a.audio,
        outdir=a.outdir,
        asr_model=a.asr_model,      
        device_choice=a.device,
        language=a.language,
        task=a.task,
        chunk_length_s=a.chunk_length_s,
        stride_left_s=a.stride_left_s,
        stride_right_s=a.stride_right_s,
    )
    print("JSON:", jpath)
    print("MD  :", mdpath)
    print("Utterances:", len(merged))


if __name__ == "__main__":
    main()
