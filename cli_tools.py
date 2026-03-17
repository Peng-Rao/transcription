import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from main import LectureProcessor
from transcription.audio_extractor import AudioExtractor
from transcription.text_processor import TextProcessor
from transcription.transcriber import Transcriber
from transcription.typst_generator import TypstGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_audio_cli(args):
    extractor = AudioExtractor()
    output = args.output or f"{Path(args.video_path).stem}_audio.wav"
    extractor.extract_audio(args.video_path, output, args.sample_rate)
    print(f"Audio extracted: {output}")


def transcribe_audio_cli(args):
    transcriber = Transcriber(
        model_path=args.whisper_cpp_model,
        models_dir=args.whisper_cpp_models_dir,
    )
    output = args.output or f"{Path(args.audio_path).stem}.srt"
    transcriber.transcribe_to_srt(args.audio_path, output, language=args.language)

    if args.text_only:
        print(transcriber.extract_text_from_srt(output))
    else:
        print(f"Transcription complete: {output}")


def process_text_cli(args):
    processor = TextProcessor()
    source_path = Path(args.input_path)

    if source_path.suffix.lower() == ".srt":
        transcriber = Transcriber(
            model_path=args.whisper_cpp_model,
            models_dir=args.whisper_cpp_models_dir,
        )
        raw_text = transcriber.extract_text_from_srt(str(source_path))
    else:
        raw_text = source_path.read_text(encoding="utf-8")

    processed = processor.process_text(raw_text)
    reduced = processor.reduce_for_llm(processed, max_sentences=args.max_sentences)

    output = args.output or f"{source_path.stem}_llm_input.txt"
    Path(output).write_text(reduced, encoding="utf-8")
    print(f"Reduced text saved: {output}")


def generate_typst_cli(args):
    load_dotenv()
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")

    text = Path(args.text_path).read_text(encoding="utf-8")
    output = args.output or f"{Path(args.text_path).stem}_notes.typ"

    generator = TypstGenerator(api_key=api_key, model=args.model)
    generator.generate_notes(text, output, args.title)
    print(f"Typst notes generated: {output}")


def batch_process_cli(args):
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    files = sorted(input_dir.glob("*.mp4"))
    if not files:
        print(f"No MP4 files found in: {input_dir}")
        return

    processor = LectureProcessor(
        api_key=api_key,
        whisper_cpp_model=args.whisper_cpp_model,
        whisper_cpp_models_dir=args.whisper_cpp_models_dir,
        llm_model=args.model,
    )

    success = 0
    for file in files:
        try:
            result = processor.process_lecture(
                str(file),
                output_dir=str(output_dir),
                keep_intermediates=args.keep_intermediates,
                language=args.language,
                max_sentences=args.max_sentences,
            )
            print(f"Done: {result}")
            success += 1
        except Exception as exc:
            logger.error(f"Failed for {file.name}: {exc}")

    print(f"Batch completed: {success}/{len(files)} succeeded")


def build_parser():
    parser = argparse.ArgumentParser(description="Transcription workflow tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_extract = subparsers.add_parser(
        "extract-audio", help="Extract WAV audio from MP4"
    )
    p_extract.add_argument("video_path")
    p_extract.add_argument("-o", "--output")
    p_extract.add_argument("--sample-rate", type=int, default=16000)
    p_extract.set_defaults(func=extract_audio_cli)

    p_transcribe = subparsers.add_parser(
        "transcribe-audio", help="Create SRT using whisper.cpp"
    )
    p_transcribe.add_argument("audio_path")
    p_transcribe.add_argument(
        "--whisper-cpp-model", default=os.getenv("WHISPER_CPP_MODEL", "base.en")
    )
    p_transcribe.add_argument(
        "--whisper-cpp-models-dir", default=os.getenv("WHISPER_CPP_MODELS_DIR")
    )
    p_transcribe.add_argument("-o", "--output")
    p_transcribe.add_argument("--language")
    p_transcribe.add_argument("--text-only", action="store_true")
    p_transcribe.set_defaults(func=transcribe_audio_cli)

    p_process = subparsers.add_parser(
        "process-text", help="NLP cleanup and token reduction"
    )
    p_process.add_argument("input_path", help="Input .txt or .srt")
    p_process.add_argument(
        "--whisper-cpp-model", default=os.getenv("WHISPER_CPP_MODEL", "base.en")
    )
    p_process.add_argument(
        "--whisper-cpp-models-dir", default=os.getenv("WHISPER_CPP_MODELS_DIR")
    )
    p_process.add_argument("--max-sentences", type=int, default=120)
    p_process.add_argument("-o", "--output")
    p_process.set_defaults(func=process_text_cli)

    p_generate = subparsers.add_parser(
        "generate-typst", help="Generate Typst note file from text"
    )
    p_generate.add_argument("text_path")
    p_generate.add_argument("-o", "--output")
    p_generate.add_argument("--title", default="Lecture Notes")
    p_generate.add_argument("--model", default="deepseek-reasoner")
    p_generate.add_argument("--api-key")
    p_generate.set_defaults(func=generate_typst_cli)

    p_batch = subparsers.add_parser(
        "batch-process", help="Process all MP4 files in a folder"
    )
    p_batch.add_argument("input_dir")
    p_batch.add_argument("--output", default="batch_output")
    p_batch.add_argument(
        "--whisper-cpp-model",
        default=os.getenv("WHISPER_CPP_MODEL", "base.en"),
        required=False,
    )
    p_batch.add_argument(
        "--whisper-cpp-models-dir", default=os.getenv("WHISPER_CPP_MODELS_DIR")
    )
    p_batch.add_argument("--model", default="deepseek-reasoner")
    p_batch.add_argument("--language")
    p_batch.add_argument("--max-sentences", type=int, default=120)
    p_batch.add_argument("--keep-intermediates", action="store_true")
    p_batch.set_defaults(func=batch_process_cli)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as exc:
        logger.error(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
