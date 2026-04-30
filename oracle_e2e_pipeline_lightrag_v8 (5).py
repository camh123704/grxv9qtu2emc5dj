#!/usr/bin/env python
"""
End-to-end offline meeting pipeline.

Takes a single media file as input (video or audio) and runs:
1. FFmpeg:         input_file -> 16 kHz mono WAV
2. Pyannote:       diarization -> segments.json (speaker-labeled segments, no words)
3. Whisper (HF):   diarized ASR -> meeting_transcript.json (with word timings)
4. Strip words:    meeting_transcript_nowords.json (no `words` field)
5. LLM summary:    meeting_ops_brief.md via any OpenAI-compatible /v1/chat/completions endpoint

This script is fully self-contained: no imports from other local .py files.
"""

from __future__ import annotations

import os
import json
import json as pyjson
import time
import argparse
import shutil
import subprocess
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
import librosa
from tqdm import tqdm
import torch
import requests
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline as hf_pipeline,
)
from transformers.utils import logging as hf_logging
from pyannote.audio import Pipeline as PyannotePipeline

# =============== Audio Extraction (FFmpeg) ===============


PIPELINE_VERSION = "v6 (Approach B default: send FULL transcript + retrieved LightRAG context to local Ollama)"

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_audio_to_wav(
    input_file: str,
    output_wav: str = "meeting.wav",
    sr: int = 16000,
    mono: bool = True,
    normalize: bool = False,
    start: float | None = None,
    end: float | None = None,
) -> str:
    """
    Extract audio from a video/audio file to a 16 kHz mono PCM WAV.
    """
    if not have_ffmpeg():
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and retry.")

    in_path = Path(input_file)
    out_path = Path(output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    if start is not None:
        cmd += ["-ss", str(float(start))]
    if end is not None:
        cmd += ["-to", str(float(end))]

    cmd += ["-i", str(in_path)]

    # drop video
    cmd += ["-vn"]

    af = []
    if normalize:
        af.append("loudnorm=I=-16:LRA=11:TP=-1.5")

    if af:
        cmd += ["-af", ",".join(af)]

    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(int(sr))]

    cmd += ["-c:a", "pcm_s16le", str(out_path)]

    print("[convert] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[convert] Wrote {out_path.resolve()}")
    return str(out_path)


# =============== Diarization (Pyannote) ===============

warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*std\(\): degrees of freedom is <= 0.*",
)

def run_diarization(
    audio_path: str,
    outdir: str,
    hf_home: Optional[str] = None,
    model_id: str = "pyannote/speaker-diarization-3.1",
    merge_gap: float = 0.25,
    segments_name: str = "segments.json",
) -> str:
    """
    Run speaker diarization on `audio_path` and write out `segments_name` in `outdir`.

    segments.json format:
    [
      {"start": float, "end": float, "speaker": "Speaker_0"},
      ...
    ]

    merge_gap: maximum allowed silence (seconds) between same-speaker segments
               to merge them into one.
    """
    AUDIO = Path(audio_path)
    OUTDIR = Path(outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if hf_home:
        os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not AUDIO.exists():
        raise FileNotFoundError(f"Audio not found: {AUDIO}")

    print("[diar] Loading pyannote pipeline:", model_id)
    pipeline = PyannotePipeline.from_pretrained(model_id)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    if device.type == "cuda":
        print("[diar] Device: GPU:", torch.cuda.get_device_name(0))
    else:
        print("[diar] Device: CPU")

    print("[diar] Running diarization on:", AUDIO)
    try:
        diar = pipeline(str(AUDIO))
    except TypeError:
        diar = pipeline({"audio": str(AUDIO)})

    segments: List[Dict[str, Any]] = []
    spk_map: Dict[str, str] = {}

    for segment, _, label in diar.itertracks(yield_label=True):
        spk = str(label)
        if spk not in spk_map:
            spk_map[spk] = f"Speaker_{len(spk_map)}"
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": spk_map[spk],
            }
        )

    # sort + merge contiguous same-speaker segments within `merge_gap` seconds
    segments.sort(key=lambda s: s["start"])
    merged: List[Dict[str, Any]] = []
    for s in segments:
        if (
            merged
            and merged[-1]["speaker"] == s["speaker"]
            and s["start"] - merged[-1]["end"] <= merge_gap
        ):
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(s)

    out_path = OUTDIR / segments_name
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    print("[diar] Speakers inferred:", sorted(set(s["speaker"] for s in merged)))
    print("[diar] Segments written to:", out_path.resolve())
    print("[diar] Total segments:", len(merged))
    return str(out_path)


# =============== Transcription (Whisper via Transformers) ===============

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*input name `inputs` is deprecated.*input_features.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*passed task=transcribe.*forced_decoder_ids.*ignored.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*attention mask is not set.*pad token is same as eos token.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*WhisperModel is using WhisperSdpaAttention.*falling back to the manual attention.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Whisper did not predict an ending timestamp.*",
)

hf_logging.set_verbosity_error()


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


def load_segments(outdir: str, segments_name: str = "segments.json") -> List[Dict[str, Any]]:
    seg_path = os.path.join(outdir, segments_name)
    if not os.path.exists(seg_path):
        # Fallback: one big segment
        return [{"start": 0.0, "end": None, "speaker": "Speaker_0"}]
    with open(seg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_asr(
    asr_model: str,
    device_choice: str = "auto",
    language: str = "auto",
    task: str = "transcribe",
    chunk_length_s: Optional[int] = None,
    stride_left_s: Optional[int] = None,
    stride_right_s: Optional[int] = None,
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

    generate_kwargs: Dict[str, Any] = {}
    if language and language != "auto":
        generate_kwargs["language"] = language
    if task:
        generate_kwargs["task"] = task

    kwargs: Dict[str, Any] = dict(
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
    segments_name: str = "segments.json",
    transcript_json_name: str = "meeting_transcript.json",
    transcript_md_name: str = "meeting_transcript.md",
    utter_merge_gap: float = 0.5,
) -> Tuple[str, str, List[Utterance]]:
    """
    Run Whisper diarized ASR using precomputed segments_name in `outdir`.
    Returns (json_path, md_path, merged_utterances).

    utter_merge_gap: max allowed gap (s) between same-speaker utterances to merge.
    """
    os.makedirs(outdir, exist_ok=True)
    y, sr = load_audio_16k_mono(audio_path, target_sr=16000)
    segs = load_segments(outdir, segments_name=segments_name)

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
            words.append(
                {
                    "word": (w.get("text") or "").strip(),
                    "start": s0 + float(ts[0]),
                    "end": s0 + float(ts[1]),
                }
            )

        utterances.append(
            Utterance(
                start=s0,
                end=s1,
                speaker=str(seg.get("speaker", "Speaker")),
                text=text,
                words=words,
            )
        )

    # Merge adjacent utterances from the same speaker with small gaps
    merged: List[Utterance] = []
    for u in utterances:
        if (
            merged
            and merged[-1].speaker == u.speaker
            and u.start - merged[-1].end <= utter_merge_gap
        ):
            merged[-1].end = u.end
            merged[-1].text = (merged[-1].text + " " + u.text).strip()
            merged[-1].words.extend(u.words)
        else:
            merged.append(u)

    jpath = os.path.join(outdir, transcript_json_name)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump([asdict(u) for u in merged], f, ensure_ascii=False, indent=2)

    mdpath = os.path.join(outdir, transcript_md_name)
    with open(mdpath, "w", encoding="utf-8") as f:
        f.write("# Speaker-labeled Transcript (Whisper large-v3, offline)\n\n")
        for u in merged:
            f.write(f"**[{_ts(u.start)}–{_ts(u.end)}] {u.speaker}:** {u.text}\n\n")

    return jpath, mdpath, merged


# =============== Strip word-level timing (no_words) ===============

def strip_word_level(src: str, dst: Optional[str] = None) -> str:
    """
    Remove any `words` field from utterances in the transcript JSON.

    If dst is None, writes alongside src as *_nowords.json.
    Returns the destination path.
    """
    src_path = Path(src)
    if dst is None:
        dst_path = src_path.with_name(src_path.stem + "_nowords.json")
    else:
        dst_path = Path(dst)

    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for u in data:
            if isinstance(u, dict) and "words" in u:
                u.pop("words", None)
    elif isinstance(data, dict):
        data.pop("words", None)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[nowords] Wrote {dst_path}")
    return str(dst_path)


# =============== LLM Analysis (OpenAI-compatible) ===============

SYSTEM_PROMPT = """
You are a comprehensive and structured meeting analyst. Your job is to extract the central arguments, key themes, decisions, and meaningful details from transcripts.
You may also be given retrieved supporting context from a RAG system. Treat that context as reference material to clarify terms, provide background, and resolve ambiguities.
Do not invent facts. If the retrieved context conflicts with the transcript, prioritize the transcript and note the discrepancy briefly.
"""

USER_INSTRUCTIONS = """
Summarize the transcript using the exact format below.

You are provided two inputs:
1) RAG CONTEXT: Retrieved excerpts from supporting documents (may be incomplete, redundant, or partially irrelevant).
2) TRANSCRIPT: The primary source of truth for what was said.

Rules:
- Use the TRANSCRIPT as the primary evidence for claims about what was said, argued, decided, or concluded.
- Use the RAG CONTEXT only to:
  (a) define acronyms/terms,
  (b) add background that is explicitly supported by the context,
  (c) identify people/organizations/equipment when the transcript is unclear,
  (d) cross-check details (but do not overwrite the transcript if they conflict).
- Do not add outside knowledge, speculation, or invented details.
- If an important detail is missing from both inputs, say “Not specified in the provided materials.”
- Do not include timestamps or long quotes (no more than 1 short quote total, <20 words, only if essential).
- Define acronyms on first use (prefer RAG CONTEXT definitions when available).
- Keep the BLUF to ~5 sentences total.
- Provide the 4 most central ideas as bullets.
- Each bullet must be followed by ~3 sentences explaining that idea.

Format:

BLUF:
<5-sentence executive summary that captures the episode’s main argument, why it matters, and the key conclusion(s).>

MAJOR IDEAS:
- <Major Idea 1 title>
  <~3 sentences summarizing this idea.>
  <Definition or Doctrinal Note from retrieved context.>
- <Major Idea 2 title>
  <~3 sentences summarizing this idea.>
  <Definition or Doctrinal Note from retrieved context.>
- <Major Idea 3 title>
  <~3 sentences summarizing this idea.>
  <Definition or Doctrinal Note from retrieved context.>
- <Major Idea 4 title>
  <~3 sentences summarizing this idea.>
  <Definition or Doctrinal Note from retrieved context.>

=== END OF SUMMARY ===

RAG CONTEXT (retrieved supporting excerpts):
{rag_context}

TRANSCRIPT:
{transcript_all}

(Produce the output now, following the template above, and end with === END OF SUMMARY ===)
"""





def _infer_host_guest_from_intro_text(utterances: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Infer (host_name, guest_name) by scanning the early transcript.

    Designed for USMA podcast-style intros like:
      - "I'm Jon Amble"
      - "joined ... by Dr. Stacey Pettyjohn"
      - "my conversation with Stacey Pettyjohn"
    """
    early = " ".join((u.get("text") or "") for u in utterances[:40])

    host = None
    mh = re.search(r"\bI['’]m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", early)
    if mh:
        host = mh.group(1).strip()

    guest = None
    mg = re.search(r"\bjoined\b.*?\bby\b\s+(?:Dr\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", early)
    if mg:
        guest = mg.group(1).strip()
    else:
        mg = re.search(r"\bconversation with\b\s+(?:Dr\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", early)
        if mg:
            guest = mg.group(1).strip()

    # Re-add "Dr." if the early text clearly uses it
    if guest and re.search(r"\bDr\.?\s+" + re.escape(guest.replace("Dr. ", "")) + r"\b", early):
        if not guest.startswith("Dr."):
            guest = "Dr. " + guest.replace("Dr. ", "")

    return host, guest


def _infer_speaker_id_mapping(
    utterances: List[Dict[str, Any]],
    host_name: Optional[str],
    guest_name: Optional[str],
) -> Dict[str, str]:
    """Map diarization labels (e.g., Speaker_0) to human names."""
    mapping: Dict[str, str] = {}

    # Speaker who says "I'm <host>"
    if host_name:
        pat = re.compile(r"\bI['’]m\s+" + re.escape(host_name) + r"\b", re.IGNORECASE)
        for u in utterances[:80]:
            if pat.search(u.get("text") or ""):
                spk = u.get("speaker")
                if isinstance(spk, str):
                    mapping[spk] = host_name
                break

    # If only host is known, infer guest as the most frequent other speaker early on
    if guest_name and mapping:
        host_id = next(iter(mapping.keys()))
        counts: Dict[str, int] = {}
        for u in utterances[:400]:
            spk = u.get("speaker")
            if isinstance(spk, str):
                counts[spk] = counts.get(spk, 0) + 1
        others = sorted(((c, s) for s, c in counts.items() if s != host_id), reverse=True)
        if others:
            mapping[others[0][1]] = guest_name

    return mapping


def rename_speakers_in_transcript_json(
    transcript_json_path: str,
    out_path: Optional[str] = None,
    replace_speaker_field: bool = False,
) -> Tuple[str, Dict[str, str]]:
    """Add/replace speaker labels with inferred human names.

    - If replace_speaker_field=False (default): keeps original `speaker` (Speaker_0/1) and adds `speaker_name`.
    - If replace_speaker_field=True: overwrites `speaker` with the human name as well.

    Returns (written_path, mapping_used). If mapping can't be inferred, returns original path and empty mapping.
    """
    p = Path(transcript_json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return transcript_json_path, {}

    host, guest = _infer_host_guest_from_intro_text(data)
    mapping = _infer_speaker_id_mapping(data, host, guest)
    if not mapping:
        return transcript_json_path, {}

    for u in data:
        spk = u.get("speaker")
        if isinstance(spk, str) and spk in mapping:
            u["speaker_name"] = mapping[spk]
            if replace_speaker_field:
                u["speaker"] = mapping[spk]

    outp = Path(out_path) if out_path else p
    outp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(outp), mapping


def verify_lightrag_storage_empty_fs(working_dir: str, input_dir: Optional[str] = None) -> Dict[str, Any]:
    """Filesystem-level verification of empty LightRAG storage.

    Returns dict with:
      - ok: bool
      - working_dir_files: int
      - input_dir_files: int (if provided)
      - sample_files: up to 10 file paths
    """
    sample: List[str] = []

    wd_files: List[str] = []
    if working_dir:
        wd = Path(working_dir)
        if wd.exists():
            wd_files = [str(f) for f in wd.rglob("*") if f.is_file()]
            sample.extend(wd_files[:10])

    in_files: List[str] = []
    if input_dir:
        ind = Path(input_dir)
        if ind.exists():
            in_files = [str(f) for f in ind.rglob("*") if f.is_file()]
            if len(sample) < 10:
                sample.extend(in_files[: max(0, 10 - len(sample))])

    ok = (len(wd_files) == 0) and (len(in_files) == 0)
    return {
        "ok": ok,
        "working_dir_files": len(wd_files),
        "input_dir_files": len(in_files),
        "sample_files": sample,
    }



def load_transcript(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compact_lines(utts: List[Dict[str, Any]], per_utt_cap: int = 400) -> str:
    lines: List[str] = []
    for u in utts:
        spk = str(u.get("speaker", "Speaker"))
        txt = (u.get("text") or "").replace("\n", " ").strip()
        if per_utt_cap and len(txt) > per_utt_cap:
            txt = txt[:per_utt_cap].rstrip() + "…"
        lines.append(f"{spk}: {txt}")
    return "\n".join(lines)


class OpenAICompatClient:
    """
    Works against any server that implements /v1/chat/completions.
    Examples:
      - Ollama OpenAI-compat: http://localhost:11434/v1
      - vLLM: http://localhost:8000/v1
      - LM Studio: http://localhost:1234/v1
      - OpenAI: https://api.openai.com/v1
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 900,
        retries: int = 2,
        backoff: float = 1.6,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(0, retries)
        self.backoff = max(1.0, backoff)
        self.session = requests.Session()

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 5000,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": False,
        }

        if extra_options:
            body.update(extra_options)

        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                r = self.session.post(url, headers=headers, json=body, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()

                choices = data.get("choices") or []
                if choices and "message" in choices[0]:
                    return (choices[0]["message"].get("content") or "").strip()

                if choices and "text" in choices[0]:
                    return (choices[0].get("text") or "").strip()

                return (data.get("response") or "").strip()

            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.backoff ** attempt)
                else:
                    raise

        raise last_err or RuntimeError("Unknown HTTP error")

class LightRAGClient:
    """LightRAG Server REST client with OpenAPI-based endpoint discovery.

    Why discovery?
      LightRAG Server endpoint paths have varied across versions (e.g. /api/query vs /query).
      This client fetches /openapi.json and finds the best matching paths for:
        - insert text
        - insert file (if supported)
        - query
    """

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        timeout: int = 300,
        retries: int = 2,
        backoff: float = 1.6,
        workspace: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.workspace = workspace
        self._token: Optional[str] = None
        self._paths: Optional[Dict[str, Any]] = None

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _full_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _request_with_retries(self, method: str, url: str, **kwargs) -> requests.Response:
        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                return requests.request(method, url, timeout=self.timeout, **kwargs)
            except Exception as exc:
                last_err = exc
                if attempt < self.retries:
                    time.sleep(self.backoff ** attempt)
                else:
                    break
        raise RuntimeError(f"LightRAG request failed: {last_err}")

    def _get_openapi_paths(self) -> Dict[str, Any]:
        """Return OpenAPI paths for the LightRAG backend."""
        if self._paths is None:
            response = self._request_with_retries("GET", self._full_url("/openapi.json"), headers=self._headers())
            if response.status_code >= 400:
                raise RuntimeError(f"Failed to fetch openapi.json: HTTP {response.status_code}: {response.text[:400]}")
            spec = response.json()
            paths = spec.get("paths", {}) or {}

            only_proxy_catchall = (len(paths) == 1 and "/{path}" in paths)
            title = (spec.get("info", {}) or {}).get("title", "")
            if only_proxy_catchall or title.strip().lower() == "fastapi":
                response2 = self._request_with_retries("GET", self._full_url("/docs/openapi.json"), headers=self._headers())
                if response2.status_code >= 400:
                    raise RuntimeError(
                        "Proxy OpenAPI detected, but /docs/openapi.json fetch failed: "
                        f"HTTP {response2.status_code}: {response2.text[:400]}"
                    )
                spec = response2.json()
                paths = spec.get("paths", {}) or {}

            self._paths = paths
        return self._paths

    def _find_endpoint(
        self,
        include_terms: List[str],
        method: str = "post",
        exclude_terms: Optional[List[str]] = None,
    ) -> str:
        paths = self._get_openapi_paths()
        method = method.lower()
        include = [term.lower() for term in include_terms]
        exclude = [term.lower() for term in (exclude_terms or [])]

        candidates: List[str] = []
        for path, spec in paths.items():
            if method not in spec:
                continue
            lower = path.lower()
            if all(term in lower for term in include) and not any(term in lower for term in exclude):
                candidates.append(path)

        if not candidates:
            raise RuntimeError(
                f"Could not find endpoint for terms={include_terms}, exclude={exclude_terms}, method={method}. "
                "Try inspecting available paths from /openapi.json."
            )

        candidates.sort(key=lambda value: (len(value), value))
        return candidates[0]

    def health(self) -> bool:
        try:
            response = self._request_with_retries("GET", self._full_url("/health"), headers=self._headers())
            return response.status_code == 200
        except Exception:
            return False

    def login(self) -> str:
        response = self._request_with_retries(
            "POST",
            self._full_url("/login"),
            headers={**self._headers(), "Content-Type": "application/x-www-form-urlencoded"},
            data={"username": self.username, "password": self.password},
        )
        if response.status_code >= 400:
            raise RuntimeError(f"LightRAG /login HTTP {response.status_code}: {response.text[:400]}")

        data = response.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"LightRAG login did not return access_token: {data}")

        self._token = token
        return token

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.workspace and "workspace" not in payload:
            payload["workspace"] = self.workspace

        for attempt in range(2):
            response = self._request_with_retries(
                "POST",
                self._full_url(path),
                headers={**self._headers(), "Content-Type": "application/json"},
                data=json.dumps(payload),
            )
            if response.status_code == 401 and attempt == 0:
                self._token = None
                self.login()
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"LightRAG {path} HTTP {response.status_code}: {response.text[:400]}")
            return response.json() if response.text else {}

        raise RuntimeError("Unexpected auth failure")

    def insert_text(self, text: str) -> Dict[str, Any]:
        try:
            path = self._find_endpoint(["documents", "text"], method="post")
        except Exception:
            path = self._find_endpoint(["insert"], method="post", exclude_terms=["file"])
        return self._post_json(path, {"text": text})

    def insert_file(self, file_path: str) -> Dict[str, Any]:
        """Insert a local file into LightRAG, preferring multipart upload when available."""
        try:
            upload_path = self._find_endpoint(["documents", "upload"], method="post")
        except Exception:
            upload_path = None

        if upload_path:
            import mimetypes

            filename = os.path.basename(file_path)
            mime = mimetypes.guess_type(filename)[0] or "text/plain"
            url = self._full_url(upload_path)
            for attempt in range(2):
                with open(file_path, "rb") as handle:
                    files = {"file": (filename, handle, mime)}
                    response = self._request_with_retries(
                        "POST",
                        url,
                        headers=self._headers(),
                        files=files,
                    )
                if response.status_code == 401 and attempt == 0:
                    self._token = None
                    self.login()
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(f"LightRAG {upload_path} HTTP {response.status_code}: {response.text[:400]}")
                return response.json()

        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
        payload = {
            "text": content,
            "file_path": os.path.basename(file_path),
            "filename": os.path.basename(file_path),
        }
        try:
            path = self._find_endpoint(["documents", "text"], method="post")
        except Exception:
            path = self._find_endpoint(["insert"], method="post", exclude_terms=["file"])
        return self._post_json(path, payload)

    def scan_documents(self) -> Optional[Dict[str, Any]]:
        candidates = [
            (["documents", "scan"], "post"),
            (["documents", "build"], "post"),
            (["documents", "index"], "post"),
            (["documents", "refresh"], "post"),
            (["scan"], "post"),
        ]
        for parts, method in candidates:
            try:
                path = self._find_endpoint(parts, method=method)
            except Exception:
                continue
            try:
                return self._post_json(path, {})
            except Exception:
                continue
        return None

    def list_documents(self, page: int = 1, page_size: int = 50) -> List[Dict[str, Any]]:
        """Return document-status records known to the server."""
        try:
            path = self._find_endpoint(["documents", "paginated"], method="get", exclude_terms=["upload", "scan"])
            response = self._request_with_retries(
                "GET",
                self._full_url(path),
                headers=self._headers(),
                params={"page": page, "page_size": page_size},
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            try:
                path = self._find_endpoint(["documents", "paginated"], method="post", exclude_terms=["upload", "scan"])
                data = self._post_json(path, {"page": page, "page_size": page_size})
            except Exception:
                path = self._find_endpoint(["documents"], method="get", exclude_terms=["upload", "scan", "paginated"])
                response = self._request_with_retries("GET", self._full_url(path), headers=self._headers())
                response.raise_for_status()
                data = response.json()

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("items", "documents", "data", "results"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        return []

    def verify_empty(self, page_size: int = 50) -> Dict[str, Any]:
        docs = self.list_documents(page=1, page_size=page_size)
        return {
            "ok": len(docs) == 0,
            "doc_count": len(docs),
            "sample": docs[0] if docs else None,
        }

    def query(
        self,
        query: str,
        mode: str = "hybrid",
        only_need_context: bool = False,
        response_type: str = "Multiple Paragraphs",
        user_prompt: Optional[str] = None,
        enable_rerank: Optional[bool] = None,
    ) -> Dict[str, Any]:
        path = self._find_endpoint(["query"], method="post")
        payload_new: Dict[str, Any] = {
            "query": query,
            "mode": mode,
            "only_need_context": bool(only_need_context),
            "response_type": response_type,
            "param": {
                "mode": mode,
                "only_need_context": bool(only_need_context),
                "response_type": response_type,
            },
        }
        if user_prompt:
            payload_new["user_prompt"] = user_prompt
            payload_new["param"]["user_prompt"] = user_prompt
        if enable_rerank is not None:
            payload_new["enable_rerank"] = bool(enable_rerank)
            payload_new["param"]["enable_rerank"] = bool(enable_rerank)

        try:
            return self._post_json(path, payload_new)
        except Exception:
            payload_old: Dict[str, Any] = {
                "query": query,
                "mode": mode,
                "only_need_context": bool(only_need_context),
                "response_type": response_type,
            }
            if user_prompt:
                payload_old["user_prompt"] = user_prompt
            if enable_rerank is not None:
                payload_old["enable_rerank"] = bool(enable_rerank)
            return self._post_json(path, payload_old)


def _extract_lightrag_context(resp: Dict[str, Any]) -> str:
    """Extract retrieved chunk text from LightRAG responses."""

    def _from_key(obj: Dict[str, Any], key: str) -> str:
        if key not in obj or not obj[key]:
            return ""
        value = obj[key]
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("content") or item.get("text") or item.get("chunk") or "")
                else:
                    parts.append(str(item))
            return "\n\n".join(part for part in parts if part)
        if isinstance(value, dict):
            return pyjson.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    for key in ("retrieved_context", "context", "contexts"):
        out = _from_key(resp, key)
        if out.strip():
            return out

    ref_out = _from_key(resp, "references") or _from_key(resp, "reference")
    if ref_out.strip() and len(ref_out.strip()) > 500:
        return ref_out

    def _parse_document_chunks_from_response(text: str) -> str:
        if not isinstance(text, str) or "Document Chunks" not in text:
            return ""
        match = re.search(r"Document Chunks[\s\S]*?```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if not match:
            return ""
        block = match.group(1).strip()
        if not block:
            return ""

        chunks: List[str] = []
        try:
            parsed = pyjson.loads(block)
            if isinstance(parsed, list):
                for obj in parsed:
                    if isinstance(obj, dict):
                        content = obj.get("content") or obj.get("text") or ""
                        if content:
                            ref_id = obj.get("reference_id") or obj.get("ref_id") or ""
                            prefix = f"[ref {ref_id}] " if ref_id else ""
                            chunks.append(prefix + str(content))
                if chunks:
                    return "\n\n".join(chunks)
        except Exception:
            pass

        for line in block.splitlines():
            line = line.strip()
            if not line or not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                obj = pyjson.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                content = obj.get("content") or obj.get("text") or ""
                if content:
                    ref_id = obj.get("reference_id") or obj.get("ref_id") or ""
                    prefix = f"[ref {ref_id}] " if ref_id else ""
                    chunks.append(prefix + str(content))
        return "\n\n".join(chunks)

    data = resp.get("data") if isinstance(resp, dict) else None
    candidates = []
    if isinstance(resp, dict):
        candidates.append(resp.get("response"))
    if isinstance(data, dict):
        candidates.append(data.get("response"))

    for candidate in candidates:
        parsed = _parse_document_chunks_from_response(candidate) if isinstance(candidate, str) else ""
        if parsed.strip():
            return parsed

    if isinstance(data, dict):
        for key in ("retrieved_context", "context", "contexts"):
            out = _from_key(data, key)
            if out.strip():
                return out
        ref_out = _from_key(data, "references") or _from_key(data, "reference")
        if ref_out.strip() and len(ref_out.strip()) > 500:
            return ref_out

    return ""


def analyze_meeting_via_lightrag(
    transcript_path: str,
    outdir: str,
    lightrag_url: str,
    lightrag_user: str,
    lightrag_pass: str,
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen3:30b-a3b-instruct-2507-fp16",
    max_tokens: int = 5000,
    temperature: float = 0.2,
    top_p: float = 0.9,
    api_key: Optional[str] = None,
    lightrag_index: bool = True,
    lightrag_workspace_tag: Optional[str] = None,
    lightrag_mode: str = "hybrid",
    lightrag_only_need_context: bool = False,
    api_timeout: int = 300,
    api_retries: int = 2,
    api_backoff: float = 1.6,
    summary_md_name: str = "meeting_ops_brief.md",
    summary_json_name: str = "meeting_ops_brief.json",
) -> str:
    """Generate the ops brief through LightRAG retrieval and optional local generation."""
    os.makedirs(outdir, exist_ok=True)

    utterances = load_transcript(transcript_path)
    meeting_text = compact_lines(utterances, per_utt_cap=400)
    if len(meeting_text) < 50:
        raise ValueError("Transcript appears too short after compaction.")

    abs_outdir = os.path.abspath(outdir)
    kb_path = os.path.join(abs_outdir, "meeting_kb.txt")
    with open(kb_path, "w", encoding="utf-8") as handle:
        handle.write(meeting_text)

    client = LightRAGClient(
        base_url=lightrag_url,
        username=lightrag_user,
        password=lightrag_pass,
        timeout=api_timeout,
        retries=api_retries,
        backoff=api_backoff,
        workspace=lightrag_workspace_tag,
    )
    client.login()

    if lightrag_index:
        client.insert_file(kb_path)
        try:
            client.scan_documents()
        except Exception:
            pass
        wait_s = int(os.getenv("LIGHRAG_POST_INSERT_WAIT", "4"))
        if wait_s > 0:
            time.sleep(wait_s)

    retrieval_query = (
        "Retrieve relevant supporting context (definitions, frameworks, tradeoffs, legal/policy constraints, "
        "and cost-imposition concepts) to help summarize the attached transcript about drones, contested airspace, "
        "and counter-UAS (C-UAS). Return context passages only."
    )
    user_prompt = (
        "Return the most relevant passages (verbatim) from the indexed documents that will help an LLM produce a "
        "grounded summary. Prefer concrete definitions, decision frameworks, and key technical/operational tradeoffs."
    )

    mode_norm = (lightrag_mode or "").lower()
    resp: Dict[str, Any] = {}
    ctx = ""
    ctx_used = ""

    if mode_norm != "bypass":
        resp = client.query(
            query=retrieval_query,
            mode=lightrag_mode,
            only_need_context=lightrag_only_need_context,
            response_type="Multiple Paragraphs",
            user_prompt=user_prompt,
        )
        ctx = _extract_lightrag_context(resp)
        ctx_used = ctx

    if lightrag_only_need_context:
        if mode_norm != "bypass":
            min_ctx = int(os.getenv("MIN_RAG_CONTEXT_CHARS", "50"))
            if (not ctx.strip()) or (len(ctx.strip()) < min_ctx):
                raise RuntimeError(
                    "RAG context too small/empty while lightrag_only_need_context=True. "
                    f"len={len(ctx.strip())} < MIN_RAG_CONTEXT_CHARS={min_ctx}. "
                    "This prevents silently generating from incomplete context."
                )

        local = OpenAICompatClient(
            base_url=base_url,
            api_key=api_key,
            timeout=max(300, int(api_timeout)),
            retries=int(api_retries),
            backoff=float(api_backoff),
        )

        system = SYSTEM_PROMPT.strip()
        user = (
            USER_INSTRUCTIONS.format(
                rag_context=ctx,
                transcript_all=meeting_text,
            ).strip()
            + "\n\n"
            + "RULES (follow strictly):\n"
            + "1) TRANSCRIPT is the ONLY authority for meeting-specific facts: who said what, decisions, taskings, numbers, dates/times, units/locations, and what occurred.\n"
            + "2) RETRIEVED CONTEXT MUST be used to improve precision, not to add meeting facts.\n"
            + "   - Define acronyms on first use.\n"
            + "   - Provide short definitions of key terms or authorities when they appear.\n"
            + "   - Fix terminology to the correct doctrinal label if the transcript is ambiguous, but do not change what the meeting meant.\n"
            + "3) When using RETRIEVED CONTEXT, label it explicitly as either 'Definition (retrieved context): ...' or 'Doctrinal note (retrieved context): ...'.\n"
            + "4) If TRANSCRIPT and RETRIEVED CONTEXT conflict, prioritize TRANSCRIPT and do not correct the meeting—only note the definition or doctrine.\n"
            + "5) Do not invent speakers, facts, or citations.\n"
        )

        summary_text = local.chat(
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    else:
        summary_text = (
            resp.get("answer")
            or resp.get("response")
            or resp.get("result")
            or (resp.get("data") if isinstance(resp.get("data"), str) else "")
            or ""
        )

    if not summary_text.strip():
        summary_text = "(empty)"

    md_path = os.path.join(abs_outdir, summary_md_name)
    json_path = os.path.join(abs_outdir, summary_json_name)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(summary_text)

    payload = {
        "summary_md": summary_text,
        "lightrag": {
            "url": lightrag_url,
            "user": lightrag_user,
            "workspace": lightrag_workspace_tag,
            "mode": lightrag_mode,
            "indexed": bool(lightrag_index),
            "only_need_context": bool(lightrag_only_need_context),
        },
        "transcript_path": transcript_path,
        "kb_path": kb_path,
        "transcript_chars": len(meeting_text),
        "retrieved_context": ctx_used,
        "retrieved_context_chars": len(ctx_used),
        "raw_response": resp,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return md_path


# =============== Orchestrator: input_file -> summary ===============

def run_pipeline(
    input_file: str,
    outdir: str = "out",
    # diarization
    hf_home: Optional[str] = None,
    diar_model_id: str = "pyannote/speaker-diarization-3.1",
    diar_merge_gap: float = 0.25,
    # ASR
    asr_model: str = "/home/jovyan/Oracle_local/models/whisper-large-v3",
    device_choice: str = "auto",
    language: str = "auto",
    task: str = "transcribe",
    chunk_length_s: Optional[int] = None,
    stride_left_s: Optional[int] = None,
    stride_right_s: Optional[int] = None,
    # output filenames
    wav_name: str = "audio.wav",
    segments_name: str = "segments.json",
    transcript_json_name: str = "meeting_transcript.json",
    transcript_md_name: str = "meeting_transcript.md",
    transcript_nowords_name: str = "meeting_transcript_nowords.json",
    summary_md_name: str = "meeting_ops_brief.md",
    summary_json_name: str = "meeting_ops_brief.json",
    # LLM (direct OpenAI-compatible chat endpoint)
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen3:30b-a3b-instruct-2507-fp16",
    max_tokens: int = 5000,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_chars_per_utt: int = 400,
    api_key: Optional[str] = None,
    http_timeout: int = 900,
    http_retries: int = 2,
    http_backoff: float = 1.6,
    extra_options_json: Optional[str] = None,
    # LightRAG (optional)
    lightrag_url: Optional[str] = None,
    lightrag_user: str = "cam",
    lightrag_pass: str = "",
    lightrag_index: bool = True,
    lightrag_workspace_tag: Optional[str] = None,
    lightrag_mode: str = "hybrid",
    lightrag_only_need_context: bool = False,
    rename_speakers: bool = True,
) -> Dict[str, str]:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # 1) input_file -> WAV
    print("========== [1/4] Input file -> WAV ==========")
    wav_path = outdir_path / wav_name
    extract_audio_to_wav(
        input_file=input_file,
        output_wav=str(wav_path),
        sr=16000,
        mono=True,
        normalize=False,
    )

    # 2) WAV -> diarization segments.json
    print("========== [2/4] WAV -> Diarization ==========")
    segments_path = run_diarization(
        audio_path=str(wav_path),
        outdir=str(outdir_path),
        hf_home=hf_home,
        model_id=diar_model_id,
        merge_gap=diar_merge_gap,
        segments_name=segments_name,
    )

    # 3) diarized ASR
    print("========== [3/4] Diarized ASR (Whisper) ==========")
    jpath, mdpath, merged = transcribe_segments(
        audio_path=str(wav_path),
        outdir=str(outdir_path),
        asr_model=asr_model,
        device_choice=device_choice,
        language=language,
        task=task,
        chunk_length_s=chunk_length_s,
        stride_left_s=stride_left_s,
        stride_right_s=stride_right_s,
        segments_name=segments_name,
        transcript_json_name=transcript_json_name,
        transcript_md_name=transcript_md_name,
        utter_merge_gap=diar_merge_gap,  # shared gap for utterance merge
    )
    print("[asr] Transcript JSON:", jpath)
    print("[asr] Transcript MD  :", mdpath)
    print("[asr] Utterances    :", len(merged))

    # 3b) strip word-level timing into *_nowords.json
    jpath_nowords = strip_word_level(
        src=jpath,
        dst=str(outdir_path / transcript_nowords_name),
    )

    # 3c) best-effort speaker name normalization (Speaker_0/1 -> real names)
    transcript_for_analysis = jpath_nowords
    if rename_speakers:
        named_path = str(outdir_path / (Path(transcript_nowords_name).stem + "_named.json"))
        try:
            out_named, spk_map = rename_speakers_in_transcript_json(
                transcript_json_path=jpath_nowords,
                out_path=named_path,
                replace_speaker_field=False,  # keep Speaker_0/1, add speaker_name
            )
            if spk_map:
                transcript_for_analysis = out_named
                print("[speakers] Inferred speaker mapping:", spk_map)
                print("[speakers] Wrote named transcript:", out_named)
            else:
                print("[speakers] No mapping inferred; leaving speaker labels unchanged.")
        except Exception as e:
            print("[speakers] Speaker rename failed; continuing without rename:", e)

    # 4) LLM analysis
    print("========== [4/4] LLM Meeting Analysis ==========")
    # If LightRAG is configured, generate the ops brief through LightRAG (retrieval + LLM).
    if lightrag_url:
        summary_md_path = analyze_meeting_via_lightrag(
            transcript_path=transcript_for_analysis,
            outdir=str(outdir_path),
            lightrag_url=lightrag_url,
            lightrag_user=lightrag_user,
            lightrag_pass=lightrag_pass,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
            lightrag_index=lightrag_index,
            lightrag_workspace_tag=lightrag_workspace_tag,
            lightrag_mode=lightrag_mode,
            lightrag_only_need_context=lightrag_only_need_context,
            api_timeout=max(300, int(http_timeout)),
            api_retries=int(http_retries),
            api_backoff=float(http_backoff),
            summary_md_name=summary_md_name,
            summary_json_name=summary_json_name,
        )
    else:
        summary_md_path = analyze_meeting(
            transcript_path=transcript_for_analysis,
            outdir=str(outdir_path),
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_chars_per_utt=max_chars_per_utt,
            api_key=api_key,
            http_timeout=http_timeout,
            http_retries=http_retries,
            http_backoff=http_backoff,
            extra_options_json=extra_options_json,
            summary_md_name=summary_md_name,
            summary_json_name=summary_json_name,
        )

    return {
        "audio_wav": str(wav_path),
        "segments_json": segments_path,
        "transcript_json": jpath,
        "transcript_nowords_json": jpath_nowords,
        "transcript_md": mdpath,
        "summary_md": summary_md_path,
    }


# =============== CLI ===============

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="End-to-end offline meeting pipeline: input_file -> diarized transcript -> LLM summary"
    )

    # Core inputs
    ap.add_argument(
        "--input_file",
        required=True,
        help="Input media file (video or audio: mp4, mkv, mp3, wav, etc.)",
    )
    ap.add_argument(
        "--asr_model",
        required=True,
        help="LOCAL FOLDER of Whisper model (e.g., /home/jovyan/Oracle_local/models/whisper-large-v3)",
    )

    # Output dir
    ap.add_argument("--outdir", default="out", help="Output directory for artifacts")

    # Diarization
    ap.add_argument("--hf_home", default=None, help="HF_HOME for pyannote cache (optional)")
    ap.add_argument(
        "--diar_model_id",
        default="pyannote/speaker-diarization-3.1",
        help="Pyannote diarization model id",
    )
    ap.add_argument(
        "--diar_merge_gap",
        type=float,
        default=0.25,
        help="Seconds allowed between same-speaker segments/utterances to merge them (e.g., 1.0)",
    )

    # ASR options
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Where to run the ASR model",
    )
    ap.add_argument("--language", default="auto", help='ISO code like "en" or "auto" to detect')
    ap.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Transcribe (same language) or translate to English",
    )
    ap.add_argument(
        "--chunk_length_s",
        type=int,
        default=None,
        help="Chunk length for long segments (e.g., 30)",
    )
    ap.add_argument(
        "--stride_left_s",
        type=int,
        default=None,
        help="Left stride seconds (e.g., 5)",
    )
    ap.add_argument(
        "--stride_right_s",
        type=int,
        default=None,
        help="Right stride seconds (e.g., 2)",
    )

    # Output filenames
    ap.add_argument("--wav_name", default="meeting.wav", help="WAV filename inside outdir")
    ap.add_argument(
        "--segments_name",
        default="segments.json",
        help="Diarization segments JSON filename inside outdir",
    )
    ap.add_argument(
        "--transcript_json_name",
        default="meeting_transcript.json",
        help="Transcript JSON filename inside outdir",
    )
    ap.add_argument(
        "--transcript_md_name",
        default="meeting_transcript.md",
        help="Transcript Markdown filename inside outdir",
    )
    ap.add_argument(
        "--transcript_nowords_name",
        default="meeting_transcript_nowords.json",
        help="Transcript JSON filename without word-level timings",
    )
    ap.add_argument(
        "--summary_md_name",
        default="meeting_ops_brief.md",
        help="LLM summary Markdown filename",
    )
    ap.add_argument(
        "--summary_json_name",
        default="meeting_ops_brief.json",
        help="LLM summary metadata JSON filename",
    )

    # LLM options
    ap.add_argument(
        "--base_url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible base URL, e.g. http://localhost:11434/v1 for Ollama",
    )
    ap.add_argument(
        "--model",
        default="qwen3:30b-a3b-instruct-2507-fp16",
        help="Model name on that endpoint (e.g., an Ollama model tag)",
    )
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    ap.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    ap.add_argument("--max_tokens", type=int, default=5000, help="Max tokens for summary")
    ap.add_argument(
        "--max_chars_per_utt",
        type=int,
        default=400,
        help="Cap per-utterance text length in prompt",
    )

    ap.add_argument(
        "--api_key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (if your endpoint requires one)",
    )
    ap.add_argument("--http_timeout", type=int, default=900)
    ap.add_argument("--http_retries", type=int, default=2)
    ap.add_argument("--http_backoff", type=float, default=1.6)
    ap.add_argument(
        "--extra_options_json",
        default=None,
        help='JSON string with provider-specific options (e.g. {"stop": ["=== END OF SUMMARY ==="]})',
    )

    # -------- LightRAG (optional) --------
    ap.add_argument(
        "--lightrag_url",
        default=None,
        help="If set, use LightRAG Server for the final ops-brief generation (e.g., http://127.0.0.1:9622).",
    )
    ap.add_argument("--lightrag_user", default="cam", help="LightRAG username (for /login).")
    ap.add_argument("--lightrag_pass", default="", help="LightRAG password (for /login).")
    ap.add_argument(
        "--lightrag_index",
        action="store_true",
        default=True,
        help="Index the current meeting transcript into LightRAG before querying (default: enabled).",
    )
    ap.add_argument(
        "--no_lightrag_index",
        dest="lightrag_index",
        action="store_false",
        help="Do NOT index the current meeting transcript into LightRAG; just query whatever is already indexed.",
    )
    ap.add_argument(
        "--lightrag_workspace_tag",
        default=None,
        help="Optional workspace tag for data isolation (works only if your server supports workspace per request).",
    )
    ap.add_argument(
        "--lightrag_mode",
        default="hybrid",
        choices=["naive", "local", "global", "hybrid", "mix", "bypass"],
        help="LightRAG query retrieval mode.",
    )
    ap.add_argument(
        "--lightrag_only_need_context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to run in context-only retrieval mode (Approach B). When enabled, the pipeline will:"
            " (1) index/query LightRAG for retrieved passages, then (2) send the FULL transcript + retrieved context"
            " to the local Ollama LLM for generation. Use --no-lightrag_only_need_context to allow server-side generation."
        ),
    )


    # -------- Post-processing --------
    ap.add_argument(
        "--rename_speakers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Best-effort: infer real speaker names from the episode intro (e.g., 'I'm Jon Amble' / 'joined by Dr. ...') "
            "and add a `speaker_name` field in the *_nowords transcript JSON. Use --no-rename_speakers to disable."
        ),
    )

    # -------- LightRAG verification (optional) --------
    ap.add_argument(
        "--lightrag_verify_empty",
        action="store_true",
        help=(
            "Verify LightRAG is empty (no ingested documents) via the server API. "
            "If --lightrag_working_dir is provided, also verify on-disk storage is empty. Exits after reporting."
        ),
    )
    ap.add_argument(
        "--lightrag_working_dir",
        default=None,
        help="Optional: local path to LightRAG working-dir (rag_storage) for filesystem empty verification.",
    )
    ap.add_argument(
        "--lightrag_input_dir",
        default=None,
        help="Optional: local path to LightRAG input-dir (inputs) for filesystem empty verification.",
    )


    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    # Optional: verify LightRAG emptiness (no docs) and exit.
    if args.lightrag_verify_empty:
        if not args.lightrag_url:
            raise SystemExit("--lightrag_verify_empty requires --lightrag_url")
        client = LightRAGClient(
            base_url=args.lightrag_url,
            username=args.lightrag_user,
            password=args.lightrag_pass,
            timeout=max(60, int(args.http_timeout)),
            retries=int(args.http_retries),
            backoff=float(args.http_backoff),
            workspace=args.lightrag_workspace_tag,
        )
        api_res = client.verify_empty(page_size=50)
        print("========== LightRAG Empty Verification (API) ==========")
        print("ok:", api_res.get("ok"))
        print("doc_count:", api_res.get("doc_count"))
        if api_res.get("sample") is not None:
            print("sample:", api_res.get("sample"))

        fs_ok = None
        if args.lightrag_working_dir or args.lightrag_input_dir:
            print("========== LightRAG Empty Verification (Filesystem) ==========")
            fs_res = verify_lightrag_storage_empty_fs(
                working_dir=args.lightrag_working_dir or "",
                input_dir=args.lightrag_input_dir,
            )
            fs_ok = bool(fs_res.get("ok"))
            print("ok:", fs_ok)
            print("working_dir_files:", fs_res.get("working_dir_files"))
            print("input_dir_files:", fs_res.get("input_dir_files"))
            if fs_res.get("sample_files"):
                print("sample_files:", fs_res.get("sample_files"))

        overall_ok = bool(api_res.get("ok")) and (fs_ok is None or fs_ok is True)
        if not overall_ok:
            raise SystemExit(1)
        return

    outputs = run_pipeline(
        input_file=args.input_file,
        outdir=args.outdir,
        hf_home=args.hf_home,
        diar_model_id=args.diar_model_id,
        diar_merge_gap=args.diar_merge_gap,
        asr_model=args.asr_model,
        device_choice=args.device,
        language=args.language,
        task=args.task,
        chunk_length_s=args.chunk_length_s,
        stride_left_s=args.stride_left_s,
        stride_right_s=args.stride_right_s,
        wav_name=args.wav_name,
        segments_name=args.segments_name,
        transcript_json_name=args.transcript_json_name,
        transcript_md_name=args.transcript_md_name,
        transcript_nowords_name=args.transcript_nowords_name,
        summary_md_name=args.summary_md_name,
        summary_json_name=args.summary_json_name,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_chars_per_utt=args.max_chars_per_utt,
        api_key=args.api_key,
        http_timeout=args.http_timeout,
        http_retries=args.http_retries,
        http_backoff=args.http_backoff,
        extra_options_json=args.extra_options_json,
        lightrag_url=args.lightrag_url,
        lightrag_user=args.lightrag_user,
        lightrag_pass=args.lightrag_pass,
        lightrag_index=args.lightrag_index,
        lightrag_workspace_tag=args.lightrag_workspace_tag,
        lightrag_mode=args.lightrag_mode,
        lightrag_only_need_context=args.lightrag_only_need_context,
        rename_speakers=args.rename_speakers,
    )

    print("\n========== DONE ==========")
    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
