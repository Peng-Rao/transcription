# Transcription: MP4 to Typst Notes

Convert lecture videos into clean, structured Typst notes.

This repository provides a full pipeline:

1. Input an MP4 file
2. Extract audio with FFmpeg
3. Transcribe with whisper.cpp through pywhispercpp
4. Clean and compress transcript text for lower token usage
5. Generate polished Typst notes with an LLM

## Features

- End-to-end MP4 to Typst workflow
- whisper.cpp inference through pywhispercpp (CoreML-ready on macOS)
- Automatic subtitle generation in SRT format
- NLP text cleanup and token reduction before LLM calls
- Typst output that is easier to version and compile than PDF-first pipelines
- CLI for single file and batch processing

## Project Structure

- main.py: main pipeline entry point
- cli_tools.py: utility subcommands
- transcription/audio_extractor.py: FFmpeg audio extraction
- transcription/transcriber.py: pywhispercpp transcription
- transcription/text_processor.py: text cleanup and token reduction
- transcription/typst_generator.py: LLM prompt + Typst generation
- transcription/prompt_template_typst.txt: Typst prompt template
- transcription/template_typst.typ: fallback Typst template

## Requirements

- Python 3.12+
- FFmpeg installed and available in PATH
- pywhispercpp (with CoreML build flags if desired)

## Installation

### 1) Create environment

```bash
uv venv
source .venv/bin/activate
```

### 2) Install pywhispercpp

For macOS CoreML support:

```bash
WHISPER_COREML=1 uv add git+https://github.com/absadiki/pywhispercpp
```

Then install project dependencies:

```bash
uv pip install .
```

### 3) Install FFmpeg

macOS:

```bash
brew install ffmpeg
```

## Configuration

Set environment variables in a .env file or shell:

```bash
DEEPSEEK_API_KEY=your_deepseek_key
WHISPER_CPP_MODEL=base.en
WHISPER_CPP_MODELS_DIR=/absolute/path/to/model-cache
```

Notes:

- WHISPER_CPP_MODEL is optional at runtime and defaults to base.en
- Models are auto-downloaded by pywhispercpp if needed
- WHISPER_CPP_MODELS_DIR is optional

## Quick Start

```bash
python main.py /path/to/lecture.mp4 --language en -o output
```

Typical output artifacts:

- lecture_subtitles.srt
- lecture_processed.txt
- lecture_llm_input.txt
- lecture_notes.typ

## Main CLI Options

```bash
python main.py /path/to/lecture.mp4 \
  --whisper-cpp-model base.en \
  --whisper-cpp-models-dir /path/to/model-cache \
  --language en \
  --max-sentences 120 \
  --llm-model deepseek-reasoner \
  --keep-intermediates \
  -o output
```

## Utility Commands

```bash
# Extract audio
python cli_tools.py extract-audio video.mp4

# Transcribe to SRT
python cli_tools.py transcribe-audio audio.wav

# Process text for LLM
python cli_tools.py process-text transcript.srt --max-sentences 100

# Generate Typst from prepared text
python cli_tools.py generate-typst reduced.txt --title "Lecture Notes"

# Batch process all mp4 files in a folder
python cli_tools.py batch-process ./videos
```

## How It Works

1. FFmpeg extracts mono speech-friendly WAV from MP4
2. pywhispercpp transcribes audio and saves SRT subtitles
3. NLP processing removes filler noise and improves readability
4. Token reduction keeps informative content for LLM efficiency
5. LLM returns Typst source
6. Sanitization pass fixes common Typst issues before save

## Troubleshooting

### pywhispercpp build issues

- Reinstall with explicit flags:

```bash
WHISPER_COREML=1 uv add --reinstall git+https://github.com/absadiki/pywhispercpp
```

### FFmpeg not found

- Confirm installation:

```bash
ffmpeg -version
```

### Typst compile errors from generated notes

- The generator includes a sanitizer for common LLM syntax issues.
- If you still hit edge cases, regenerate once or adjust transcription/prompt quality.

## Roadmap

- Stronger Typst lint/syntax normalization
- Optional local model backends for note generation
- Better chapter/topic segmentation for long lectures

## License

MIT (or your preferred license)
