import os
import shlex
import subprocess
import re
import time
import base64
from pathlib import Path

import streamlit as st


# -------------------------
# Fixed defaults (not user-editable)
# -------------------------
DEFAULT_PYTHON = "/home/jovyan/venvs/hfmeet/bin/python"
DEFAULT_SCRIPT = "/home/jovyan/Oracle_local/oracle_e2e_pipeline_lightrag_v8.py"

DEFAULT_ASR_MODEL = "/home/jovyan/Oracle_local/models/whisper-large-v3"
DEFAULT_INPUT_DIR = "/home/jovyan/Oracle_local/Podcasts"
DEFAULT_OUTROOT = "/home/jovyan/Oracle_local/out"

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LIGHTRAG_URL = "http://127.0.0.1:9621"  # fixed, not shown in UI

UPLOAD_DIR = Path("/home/jovyan/Oracle_local/ui_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Oracle + LightRAG Runner", layout="wide")

if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

with open("/home/jovyan/Oracle_local/Oracle_Logo.png", "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode("utf-8")

st.markdown(
    """
    <style>
    .logo-static img {
        display: block;
    }

    .logo-pulse img {
        display: block;
        animation: oraclePulse 1.8s ease-in-out infinite;
        border-radius: 18px;
    }

    @keyframes oraclePulse {
        0% {
            transform: scale(1);
            filter: drop-shadow(0 0 6px rgba(0, 200, 255, 0.20))
                    drop-shadow(0 0 12px rgba(0, 200, 255, 0.15));
        }
        50% {
            transform: scale(1.04);
            filter: drop-shadow(0 0 14px rgba(0, 200, 255, 0.55))
                    drop-shadow(0 0 28px rgba(0, 200, 255, 0.35));
        }
        100% {
            transform: scale(1);
            filter: drop-shadow(0 0 6px rgba(0, 200, 255, 0.20))
                    drop-shadow(0 0 12px rgba(0, 200, 255, 0.15));
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #001B2D;
    }

    header[data-testid="stHeader"] {
        background-color: #000814;
    }

    section[data-testid="stSidebar"] {
        background-color: #000814;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #000814;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    h1 {
        margin-bottom: 0.1rem !important;
    }

    h2 {
        margin-top: 0.4rem !important;
    }

    div.block-container {
        padding-top: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def safe_name(name: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    out = "".join(c for c in name if c in keep).strip("._-")
    return (out[:180] if out else "")


def run_and_stream(cmd, env=None, cwd=None):
    """Run subprocess and yield output lines."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    try:
        for line in iter(proc.stdout.readline, ""):
            yield line
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
    return proc.returncode


def list_artifacts(outdir: Path):
    if not outdir.exists():
        return []
    files = sorted(
        [p for p in outdir.rglob("*") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files


def mask_cmd(cmd_list):
    """Mask password in command preview (display only)."""
    masked = cmd_list[:]
    for i, tok in enumerate(masked):
        if tok == "--lightrag_pass" and i + 1 < len(masked):
            masked[i + 1] = "********"
    return masked


# -------------------------
# Hardcoded LightRAG creds (placeholders)
# -------------------------
LIGHTRAG_USER = "cam"
LIGHTRAG_PASS = "StrongPass123"

# -------------------------
# Session state: Run tag blank by default, persists across reruns
# -------------------------
if "run_tag" not in st.session_state:
    st.session_state.run_tag = ""

# -------------------------
# UI
# -------------------------
top_left, top_right = st.columns([4, 1.5])

with top_left:
    st.title("Oracle Meeting Summarization Tool")

with top_right:
    logo_class = "logo-pulse" if st.session_state.get("pipeline_running", False) else "logo-static"
    st.markdown(
        f"""
        <div class="{logo_class}">
            <img src="data:image/png;base64,{logo_b64}" width="320">
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("Input")

    input_mode = st.radio("Input source", ["File Upload", "Use existing path"], index=0)

    input_path = None
    if input_mode == "File Upload":
        up = st.file_uploader("Upload audio/video", type=None)
        if up:
            fname = safe_name(up.name) or "upload"
            input_path = UPLOAD_DIR / fname
            input_path.write_bytes(up.getbuffer())
            st.success(f"Saved: {input_path}")
    else:
        input_path_str = st.text_input(
            "Input file path",
            value=f"{DEFAULT_INPUT_DIR}/Airspace_Violations_and_Drones.mp3",
        )
        input_path = Path(input_path_str)

    run_tag = st.text_input("Run tag", key="run_tag")
    run_tag_safe = safe_name(run_tag) if run_tag else ""

    outdir = Path(DEFAULT_OUTROOT) / run_tag_safe if run_tag_safe else None
    st.caption(f"Will write to: {outdir}" if outdir else "Will write to: (set Run tag)")

    st.divider()
    st.header("Options")

    show_raw_logs = st.checkbox("Show raw logs", value=False)

    diar_merge_gap = st.number_input(
        "Diarization merge gap (sec)",
        min_value=0.0,
        max_value=10.0,
        value=1.5,
        step=0.1,
    )

    model = st.text_input("Ollama Model", value="gpt-oss:20b")

    use_lightrag = st.checkbox("Enable LightRAG", value=True)
    lightrag_mode = st.selectbox("Lightrag Mode", ["hybrid", "naive", "local", "mix"], index=0)

    lightrag_url = DEFAULT_LIGHTRAG_URL

    lightrag_user = LIGHTRAG_USER
    lightrag_pass = LIGHTRAG_PASS
    st.caption(f"LightRAG user: {lightrag_user} (password hidden)")


# -------------------------
# Derived output names from run tag
# -------------------------
if run_tag_safe:
    wav_name = f"{run_tag_safe}.wav"
    segments_name = f"segs_{run_tag_safe}.json"
    transcript_json_name = f"trans_{run_tag_safe}.json"
    transcript_md_name = f"trans_{run_tag_safe}.md"
    transcript_nowords_name = f"trans_nowords_{run_tag_safe}.json"
    summary_md_name = f"summary_{run_tag_safe}.md"
    summary_json_name = f"summary_{run_tag_safe}.json"
else:
    wav_name = segments_name = transcript_json_name = transcript_md_name = ""
    transcript_nowords_name = summary_md_name = summary_json_name = ""


col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Run")
    run_btn = st.button(
        "▶ Run pipeline",
        type="primary",
        disabled=(not input_path or str(input_path) == "" or not run_tag_safe),
    )

log_placeholder = st.empty()
log_caption = st.empty()
status_placeholder = st.empty()

stage_panel = st.empty()
progress_container = st.empty()
progress_text = st.empty()

if run_btn:
    st.session_state.pipeline_running = True
    python_bin = DEFAULT_PYTHON
    script_path = DEFAULT_SCRIPT
    asr_model = DEFAULT_ASR_MODEL
    base_url = DEFAULT_BASE_URL

    if not Path(python_bin).exists():
        st.error(f"Python not found: {python_bin}")
        st.stop()
    if not Path(script_path).exists():
        st.error(f"Script not found: {script_path}")
        st.stop()
    if not input_path.exists():
        st.error(f"Input file not found: {input_path}")
        st.stop()
    if not Path(asr_model).exists():
        st.error(f"ASR model path not found: {asr_model}")
        st.stop()

    assert outdir is not None, "outdir must be set if run_btn is enabled"
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_bin,
        script_path,
        "--input_file",
        str(input_path),
        "--asr_model",
        str(asr_model),
        "--diar_merge_gap",
        str(diar_merge_gap),
        "--wav_name",
        wav_name,
        "--segments_name",
        segments_name,
        "--transcript_json_name",
        transcript_json_name,
        "--transcript_md_name",
        transcript_md_name,
        "--transcript_nowords_name",
        transcript_nowords_name,
        "--summary_md_name",
        summary_md_name,
        "--summary_json_name",
        summary_json_name,
        "--outdir",
        str(outdir),
        "--base_url",
        base_url,
        "--model",
        model,
    ]

    if use_lightrag:
        cmd += [
            "--lightrag_url",
            lightrag_url,
            "--lightrag_user",
            lightrag_user,
            "--lightrag_pass",
            lightrag_pass,
            "--lightrag_mode",
            lightrag_mode,
        ]

    with st.expander("Command preview", expanded=False):
        st.code(" \\\n  ".join(shlex.quote(c) for c in mask_cmd(cmd)), language="bash")

    status_placeholder.info("Running…")

    if not show_raw_logs:
        log_placeholder.empty()
        log_caption.caption("Logs hidden (enable **Show raw logs** in the sidebar to view).")

    env = os.environ.copy()
    env["OLLAMA_MODEL"] = model
    env["BASE_URL"] = base_url

    # --- Stage tracking driven by pipeline headers ---
    stage_re = re.compile(r"^=+\s*\[(\d+)\s*/\s*(\d+)\]\s*(.*?)\s*=+\s*$")
    stage_idx = 0
    stage_total = 4
    stage_started_at = None
    stage_elapsed = {}  # kept internally, but not displayed

    FIXED_STAGE_LABELS = [
        "File Conversion",
        "Diarization",
        "Transcription",
        "Summarization",
    ]

    def stage_label(i: int) -> str:
        if 1 <= i <= len(FIXED_STAGE_LABELS):
            return FIXED_STAGE_LABELS[i - 1]
        return f"Stage {i}"

    def render_stage_ui():
        lines_md = []
        for i in range(1, stage_total + 1):
            label = stage_label(i)

            if stage_idx > stage_total:
                icon = "✅"
            else:
                if i < stage_idx:
                    icon = "✅"
                elif i == stage_idx:
                    icon = "🟡"
                else:
                    icon = "⬜"

            # Timers removed: no dur_txt appended
            lines_md.append(f"{icon} **{i}/{stage_total}** {label}")

        stage_panel.markdown("\n\n".join(lines_md) if lines_md else "Waiting for pipeline output…")

        frac = 0.0
        if stage_total > 0:
            if stage_idx <= 0:
                frac = 0.0
            elif stage_idx > stage_total:
                frac = 1.0
            else:
                frac = (stage_idx - 1) / stage_total
        progress_container.progress(frac)

        if stage_idx <= 0:
            progress_text.caption("Waiting for pipeline output…")
        elif stage_idx > stage_total:
            progress_text.caption("Completed.")
        else:
            progress_text.caption(f"Current: Step {stage_idx}/{stage_total} — {stage_label(stage_idx)}")

    asr_pct_re = re.compile(r"ASR by segment:\s*(\d+)%")

    lines = []
    for line in run_and_stream(cmd, env=env, cwd=str(Path(script_path).parent)):
        clean = line.replace("\r", "")

        m_stage = stage_re.match(clean.strip())
        if m_stage:
            new_idx = int(m_stage.group(1))
            new_total = int(m_stage.group(2))

            # Keep elapsed times internally (even though we don't display them)
            if stage_idx and stage_started_at and stage_idx <= stage_total:
                stage_elapsed[stage_idx] = time.time() - stage_started_at

            stage_idx = new_idx
            stage_total = new_total

            stage_started_at = time.time()
            render_stage_ui()

            if show_raw_logs:
                log_caption.empty()
                lines.append(clean)
                log_placeholder.code("".join(lines[-400:]), language="text")
            continue

        m_pct = asr_pct_re.search(clean)
        if m_pct:
            pct = int(m_pct.group(1))
            if stage_idx > 0 and stage_idx <= stage_total:
                progress_text.caption(
                    f"Current: Step {stage_idx}/{stage_total} — {stage_label(stage_idx)} | ASR by segment: {pct}%"
                )
            else:
                progress_text.caption(f"ASR by segment: {pct}%")
            continue

        if show_raw_logs:
            log_caption.empty()
            lines.append(clean)
            log_placeholder.code("".join(lines[-400:]), language="text")

        if stage_idx:
            render_stage_ui()

    if stage_idx and stage_started_at and stage_idx <= stage_total:
        stage_elapsed[stage_idx] = time.time() - stage_started_at

    if stage_total and stage_idx == stage_total:
        stage_idx = stage_total + 1

    render_stage_ui()
    status_placeholder.success("Done.")
    st.session_state.pipeline_running = False

    st.divider()
    st.subheader("Outputs")

    md_path = outdir / summary_md_name
    if md_path.exists():
        md_text = md_path.read_text(encoding="utf-8", errors="ignore")
        st.markdown(md_text)
        st.download_button(
            "Download summary (.md)",
            data=md_text,
            file_name=md_path.name,
            mime="text/markdown",
        )
    else:
        st.warning(f"Summary markdown not found: {md_path}")

    st.subheader("Artifacts in run folder")
    artifacts = list_artifacts(outdir)
    if artifacts:
        for p in artifacts[:50]:
            cols = st.columns([3, 1])
            cols[0].write(str(p.relative_to(outdir)))
            try:
                cols[1].download_button("Download", data=p.read_bytes(), file_name=p.name)
            except Exception:
                cols[1].write("—")
    else:
        st.info("No artifacts found yet.")