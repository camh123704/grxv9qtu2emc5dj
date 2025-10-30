# DSE-Capstone-12

## 📝 Meeting Summarizer Pipeline

This project is an end-to-end pipeline for generating summaries from recorded meeting files (e.g., `.mp4`, `.mov`). It processes raw recordings into structured, speaker-aware summaries using a combination of  tools in diarization, transcription, and large language models.

### 🔧 Pipeline Overview

1. **Audio Extraction**  
   Demuxes the audio from video files and converts it to a clean `.wav` format.

2. **Speaker Diarization**  
   Identifies and segments speakers using [pyannote-audio](https://huggingface.co/pyannote/speaker-diarization-3.1).

3. **Transcription**  
   Transcribes diarized audio using [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3).

4. **Targeted Summarization**  
   Summarizes the transcript using a [LLaMA](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) language model, optionally focusing on specific queries, aspects, or roles.

### 📂 Input Format

- Accepts video files: `.mp4`, `.mov`, `.mkv`, etc.
- Outputs:
  - `.wav` file (audio)
  - speaker-attributed transcript
  - one or more targeted summaries

Note: Above code uses downloaded models from Hugging Face. Code can be modified to utilize Hugging Face Tokens instead.
