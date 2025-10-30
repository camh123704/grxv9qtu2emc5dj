"""
Offline meeting summary using a locally saved Hugging Face LLM (Transformers).

Inputs:
- Transcript .json (e.g., {start, end, speaker, text})

Outputs (written to --outdir, default ./out):
- meeting_ops_brief.md
- meeting_ops_brief.json

Requirements:
  pip install torch transformers bitsandbytes
  LLM downloaded
"""


from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

SYSTEM_PROMPT = (
"""
You are a precise meeting analyst.
"""
)

USER_INSTRUCTIONS = (
"""
Summarize the meeting transcript in detailed bullets organized by topic with sub-bullets.

Rules:
- Detect and group content into 10-15 topics (titles in ALL CAPS).
- Under each topic, use detailed sub-bullets (one idea per bullet).
- If roles are inferable, replace "Speaker_X" with a likely role (e.g., "Moderator", "KACH rep"); otherwise keep the tag.
- Do not include timestamps, or long quotes.

Structure:
# EXECUTIVE SUMMARY (5–10 bullets, no sub-bullets)
# DETAILED SUMMARY BY TOPIC
- TOPIC NAME
  - sub-bullet
  - sub-bullet
  - (etc.)

End your output with the line: === END OF SUMMARY ===.

Transcript:
{transcript_all}

=== END OF PROMPT ===
"""
)

STOP_MARKER = "=== END OF SUMMARY ==="

TARGET_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ts(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def load_transcript(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compact_transcript(utts: List[Dict[str, Any]], max_chars_per_utt: int = 600):
    lines = []
    for u in utts:
        start = ts(float(u.get("start", 0)))
        spk = str(u.get("speaker", "Speaker"))
        txt = (u.get("text") or "").replace("\n", " ").strip()
        if max_chars_per_utt is not None and len(txt) > max_chars_per_utt:
            txt = txt[:max_chars_per_utt].rstrip() + "…"
        lines.append(f"[{start}] {spk}: {txt}")
    return "\n".join(lines)

def _approx_tokens(chars: int) -> int:
    return max(1, chars // 4)

def _load_llm(
    model_path: str,
    load_in_4bit: bool = False,
    dtype: str = "fp16",
    rope_factor: Optional[float] = None,
):
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model path is not a directory: {model_path}\n"
            "Pass the local folder that contains config.json and model*.safetensors."
        )

    required_any = [
        ("config.json",),
        ("tokenizer.json", "tokenizer.model"),  
    ]
    missing = []
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        missing.append("config.json")
    if not (os.path.isfile(os.path.join(model_path, "tokenizer.json")) or
            os.path.isfile(os.path.join(model_path, "tokenizer.model"))):
        missing.append("tokenizer.json|tokenizer.model")
    if missing:
        raise FileNotFoundError(
            f"Model folder missing required files: {missing}\nFolder: {model_path}"
        )

    local_only = dict(local_files_only=True)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, **local_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    device_map = "cuda:0" if TARGET_DEVICE.type == "cuda" else "cpu"

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 if TARGET_DEVICE.type == "cuda" else torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            **local_only,
        )
    else:
        if TARGET_DEVICE.type == "cuda":
            torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype, torch.float16)
        else:
            torch_dtype = torch.bfloat16 if dtype == "bf16" and hasattr(torch, "bfloat16") else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **local_only,
        )

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id

    if rope_factor and rope_factor > 1.0:
        model.config.rope_scaling = {"type": "linear", "factor": float(rope_factor)}

    return tok, model


def _apply_chat_template(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        tokenize=False,
        add_generation_prompt=True
    )

def _get_context_len(tokenizer, model) -> int:
    tml = getattr(tokenizer, "model_max_length", None)
    if isinstance(tml, int) and 0 < tml < 10_000_000:
        return tml
    cfg = getattr(model, "config", None)
    mpe = getattr(cfg, "max_position_embeddings", None) if cfg else None
    if isinstance(mpe, int) and mpe > 0:
        return mpe
    return 8192

def _squeeze_to_context(
    tokenizer,
    model,
    prompt_text: str,
    max_new_tokens: int,
    ctx_margin: int = 600
):
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    input_len = input_ids.shape[1]

    ctx = _get_context_len(tokenizer, model)

    need = input_len + max_new_tokens + 4
    if need <= ctx:
        attn = torch.ones_like(input_ids, dtype=torch.long)
        return input_ids.to(TARGET_DEVICE), attn.to(TARGET_DEVICE), False, ctx, input_len

    keep = max(256, ctx - max_new_tokens - ctx_margin)
    head = int(keep * 0.7)
    tail = keep - head

    ids = input_ids[0]
    squeezed = torch.cat([ids[:head], ids[-tail:]], dim=0).unsqueeze(0)

    note = "\n\n[Note: transcript middle truncated to fit context. Summarize faithfully without guessing or inventing details.]"
    note_ids = tokenizer(note, return_tensors="pt", add_special_tokens=False)["input_ids"]

    final_ids = torch.cat([squeezed, note_ids], dim=1)
    attn = torch.ones_like(final_ids, dtype=torch.long)

    return final_ids.to(TARGET_DEVICE), attn.to(TARGET_DEVICE), True, ctx, input_len

class EndMarkerCriteria(StoppingCriteria):
    def __init__(self, tokenizer, end_marker: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.end_ids = tokenizer(end_marker, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        self.window = len(self.end_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] < self.window:
            return False
        tail = input_ids[0, -self.window:].tolist()
        return tail == self.end_ids

def analyze_local(
    transcript_path: str,
    outdir: str,
    model_path: str,
    num_predict: int = 3000,            
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_chars_per_utt: int = 600,
    load_in_4bit: bool = False,
    dtype: str = "fp16",
    rope_factor: Optional[float] = None,
) -> str:
    os.makedirs(outdir, exist_ok=True)

    utts = load_transcript(transcript_path)
    transcript_text = compact_transcript(utts, max_chars_per_utt=max_chars_per_utt)
    user = USER_INSTRUCTIONS.format(transcript_all=transcript_text)

    tokenizer, model = _load_llm(
        model_path=model_path,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        rope_factor=rope_factor,
    )

    prompt = _apply_chat_template(tokenizer, SYSTEM_PROMPT, user)

    input_ids, attention_mask, truncated, ctx_len, prompt_token_len = _squeeze_to_context(
        tokenizer=tokenizer,
        model=model,
        prompt_text=prompt,
        max_new_tokens=num_predict,
        ctx_margin=600,
    )

    stopping_criteria = StoppingCriteriaList([EndMarkerCriteria(tokenizer, STOP_MARKER)])

    gen_kwargs = dict(
        max_new_tokens=int(num_predict),
        do_sample=(temperature > 0.0),
        temperature=float(temperature),
        top_p=float(top_p),
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.0,
        stopping_criteria=stopping_criteria,
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )

    gen_tokens = outputs[0][input_ids.shape[1]:]
    text_out = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    consolidated = text_out.strip()

    saw_end_marker = STOP_MARKER in consolidated
    stop_reason = "end_marker" if saw_end_marker else ("max_new_tokens" if gen_tokens.shape[0] >= num_predict else "other")

    md_path = os.path.join(outdir, "meeting_ops_brief.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(consolidated)

    json_path = os.path.join(outdir, "meeting_ops_brief.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": model_path,
                "input_chars": len(transcript_text),
                "approx_input_tokens_prompt_chars": len(prompt),
                "approx_input_tokens_estimate": _approx_tokens(len(prompt)),
                "actual_input_tokens_after_squeeze": int(input_ids.shape[1]),
                "model_context_window": ctx_len,
                "prompt_token_len_before_squeeze": prompt_token_len,
                "generated_new_tokens": int(gen_tokens.shape[0]),
                "saw_end_marker": saw_end_marker,
                "stop_reason": stop_reason,
                "consolidated": consolidated,
                "params": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": num_predict,
                    "max_chars_per_utt": max_chars_per_utt,
                    "load_in_4bit": load_in_4bit,
                    "dtype": dtype,
                    "rope_factor": rope_factor,
                    "device": str(TARGET_DEVICE),
                    "truncated_middle": truncated,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return md_path

# CLI

def _parse_args():
    ap = argparse.ArgumentParser(
        description="Create a consolidated summary + analysis from a meeting transcript using a locally saved HF LLM (single-pass)."
    )
    ap.add_argument("--transcript", default="out/meeting_transcript.json")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--model_path", default="meta-llama/Llama-3.2-8B-Instruct",
                    help="Local dir or HF repo id for the chat-tuned model.")
    ap.add_argument("--num_predict", type=int, default=3000, help="Max new tokens for the summary")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_chars_per_utt", type=int, default=600,
                    help="Truncate very long single turns to this many chars")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Load the model in 4-bit (bitsandbytes NF4) to save VRAM")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"],
                    help="Model dtype when not using 4-bit (fp16 on GPU; bf16/float32 on CPU).")
    ap.add_argument("--rope_factor", type=float, default=None,
                    help="Optional RoPE scaling factor (e.g., 1.5 or 2.0) to extend context; quality can degrade.")
    return ap.parse_args()

def main():
    args = _parse_args()
    md = analyze_local(
        transcript_path=args.transcript,
        outdir=args.outdir,
        model_path=args.model_path,
        num_predict=args.num_predict,
        temperature=args.temperature,
        top_p=args.top_p,
        max_chars_per_utt=args.max_chars_per_utt,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        rope_factor=args.rope_factor,
    )
    print("Wrote:", md)

if __name__ == "__main__":
    main()
