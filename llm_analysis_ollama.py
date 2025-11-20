from __future__ import annotations
import os, json, argparse, time
from typing import List, Dict, Any, Optional

import requests

SYSTEM_PROMPT = (
"""
You are a comprehensive and structured meeting analyst.
"""
)

USER_INSTRUCTIONS = (
"""
Summarize the meeting transcript in the exact structure below. Follow all rules.

Rules:
- Use only information present in the transcript.
- Detect and group content into the 10 most important topics (titles MUST be in ALL CAPS).
- Under each topic, use detailed sub-bullets.
- Replace "Speaker_X" with a role/name only if explicitly inferable from transcript context; otherwise keep the tag.
- Do NOT include timestamps or long quotes.
- End with the exact line: === END OF SUMMARY ===

Exact structure (copy these headers exactly):
# EXECUTIVE SUMMARY
- bullet
- bullet
- (5–10 bullets total, no sub-bullets)

# DETAILED SUMMARY BY TOPIC
- TOPIC NAME (ALL CAPS)
  - detailed sub-bullet
  - detailed sub-bullet
  - (continue until all details included)
- NEXT TOPIC (ALL CAPS)
  - detailed sub-bullet
  - detailed sub-bullet
  - (continue until all details included)
(continue until you have 10 topics)

Transcript:
{transcript_all}

(Produce the output now, following the template above, and end with === END OF SUMMARY ===)
"""
)

# ---------------- Utilities ----------------

def load_transcript(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compact_lines(utts: List[Dict[str, Any]], per_utt_cap: int = 400) -> str:
    lines = []
    for u in utts:
        spk = str(u.get("speaker", "Speaker"))
        txt = (u.get("text") or "").replace("\n", " ").strip()
        if per_utt_cap and len(txt) > per_utt_cap:
            txt = txt[:per_utt_cap].rstrip() + "…"
        lines.append(f"{spk}: {txt}")
    return "\n".join(lines)

# ---------------- OpenAI-Compatible Client ----------------

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
        max_tokens: int = 700,
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
            # Merge any provider-specific knobs directly into payload.
            body.update(extra_options)

        last_err = None
        for attempt in range(self.retries + 1):
            try:
                r = self.session.post(url, headers=headers, json=body, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()

                # Standard OpenAI-compatible shape
                choices = data.get("choices") or []
                if choices and "message" in choices[0]:
                    return (choices[0]["message"].get("content") or "").strip()

                # Some servers return plain "text"
                if choices and "text" in choices[0]:
                    return (choices[0].get("text") or "").strip()

                # Fallback for odd servers
                return (data.get("response") or "").strip()

            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.backoff ** attempt)
                else:
                    raise

        raise last_err or RuntimeError("Unknown HTTP error")

# ---------------- Orchestration ----------------

def analyze_meeting(
    transcript_path: str,
    outdir: str,
    base_url: str,
    model: str,
    max_tokens: int = 700,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_chars_per_utt: int = 400,
    api_key: Optional[str] = None,
    http_timeout: int = 900,
    http_retries: int = 2,
    http_backoff: float = 1.6,
    extra_options_json: Optional[str] = None,
) -> str:
    os.makedirs(outdir, exist_ok=True)

    utts = load_transcript(transcript_path)
    prompt_body = compact_lines(utts, per_utt_cap=max_chars_per_utt)
    if len(prompt_body) < 50:
        raise ValueError("Transcript appears too short after compaction.")

    extra_options = None
    if extra_options_json:
        try:
            extra_options = json.loads(extra_options_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"extra_options_json must be valid JSON. Got error: {e}")

    client = OpenAICompatClient(
        base_url=base_url,
        api_key=api_key,
        timeout=http_timeout,
        retries=http_retries,
        backoff=http_backoff,
    )

    user_prompt = USER_INSTRUCTIONS.format(transcript_all=prompt_body)

    print(f"[inference] model={model} max_tokens={max_tokens} temperature={temperature} top_p={top_p}")
    raw = client.chat(
        model=model,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_options=extra_options,
    ).strip()

    abs_outdir = os.path.abspath(outdir)
    raw_path = os.path.join(abs_outdir, "meeting_ops_brief.raw.md")
    final_path = os.path.join(abs_outdir, "meeting_ops_brief.md")

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw if raw else "(empty)")

    # No validation, no repair, no marker/header enforcement.
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(raw if raw else "(empty)")

    print("Raw  :", raw_path)
    print("Final:", final_path)
    return final_path

# ---------------- CLI ----------------

def _parse_args():
    ap = argparse.ArgumentParser(description="Structured meeting summary via any OpenAI-compatible chat endpoint.")
    ap.add_argument("--transcript", default="out/meeting_transcript.json")
    ap.add_argument("--outdir", default="out")

    ap.add_argument("--base_url", default="http://localhost:11434/v1",
                    help="OpenAI-compatible base URL, e.g. http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen3:30b-a3b-instruct-2507-fp16")

    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_chars_per_utt", type=int, default=400)

    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"),
                    help="Optional. If omitted, no Authorization header is sent.")

    ap.add_argument("--extra_options_json", default=None,
                    help="Optional JSON string merged into request body for provider-specific params.")

    ap.add_argument("--http_timeout", type=int, default=900)
    ap.add_argument("--http_retries", type=int, default=2)
    ap.add_argument("--http_backoff", type=float, default=1.6)
    return ap.parse_args()

def main():
    args = _parse_args()
    analyze_meeting(
        transcript_path=args.transcript,
        outdir=args.outdir,
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
    )

if __name__ == "__main__":
    main()
