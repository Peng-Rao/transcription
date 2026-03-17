import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from transcription.audio_extractor import AudioExtractor
from transcription.text_processor import TextProcessor
from transcription.transcriber import Transcriber
from transcription.typst_generator import TypstGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LectureProcessor:
    def __init__(
        self,
        api_key=None,
        whisper_cpp_model="base.en",
        whisper_cpp_models_dir=None,
        llm_model="deepseek-reasoner",
    ):
        self.audio_extractor = AudioExtractor()
        self.transcriber = Transcriber(
            model_path=whisper_cpp_model,
            models_dir=whisper_cpp_models_dir,
        )
        self.text_processor = TextProcessor()
        self.typst_generator = TypstGenerator(
            api_key=api_key,
            model=llm_model,
        )

    def process_lecture(
        self,
        video_path,
        output_dir="output",
        keep_intermediates=False,
        language=None,
        max_sentences=120,
    ):
        """
        Complete pipeline to process an MP4 lecture into Typst notes.

        Args:
            video_path (str): Path to the input video file
            output_dir (str): Directory to save output files
            keep_intermediates (bool): Whether to keep intermediate files
            language (str): Language code for transcription (e.g., 'en', 'es')

        Returns:
            str: Path to the generated Typst file
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if video_path.suffix.lower() != ".mp4":
            raise ValueError("Input must be an MP4 file.")

        logger.info(f"Processing lecture: {video_path.name}")

        # Step 1: Extract audio
        logger.info("Step 1: Extracting audio from video...")
        audio_path = output_dir / f"{video_path.stem}_audio.wav"
        self.audio_extractor.extract_audio(str(video_path), str(audio_path))

        # Step 2: Transcribe audio to subtitles
        logger.info("Step 2: Transcribing audio to subtitles using whisper.cpp...")
        srt_path = output_dir / f"{video_path.stem}_subtitles.srt"
        self.transcriber.transcribe_to_srt(
            str(audio_path), str(srt_path), language=language
        )

        # Step 3: Extract and process text
        logger.info("Step 3: Processing transcript text...")
        raw_text = self.transcriber.extract_text_from_srt(str(srt_path))
        processed_text = self.text_processor.process_text(raw_text)
        llm_text = self.text_processor.reduce_for_llm(
            processed_text, max_sentences=max_sentences
        )

        # Save processed text artifacts
        text_path = output_dir / f"{video_path.stem}_processed.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(processed_text)

        reduced_path = output_dir / f"{video_path.stem}_llm_input.txt"
        with open(reduced_path, "w", encoding="utf-8") as f:
            f.write(llm_text)

        # Step 4: Generate Typst notes
        logger.info("Step 4: Generating Typst notes with LLM...")
        typst_path = output_dir / f"{video_path.stem}_notes.typ"
        self.typst_generator.generate_notes(
            llm_text,
            str(typst_path),
            title=f"Lecture Notes: {video_path.stem}",
        )

        # Cleanup intermediate files if requested
        if not keep_intermediates:
            if audio_path.exists():
                audio_path.unlink()
            if srt_path.exists():
                srt_path.unlink()

        logger.info(f"Processing complete! Typst notes saved to: {typst_path}")
        return str(typst_path)


def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Convert MP4 lecture videos to Typst notes"
    )
    parser.add_argument("video_path", help="Path to the input MP4 file")
    parser.add_argument(
        "-o", "--output", default="output", help="Output directory (default: output)"
    )
    parser.add_argument(
        "-k",
        "--keep-intermediates",
        action="store_true",
        help="Keep intermediate files (audio, subtitles)",
    )

    parser.add_argument(
        "--whisper-cpp-model",
        default=os.getenv("WHISPER_CPP_MODEL", "base.en"),
        help="Model name or local model path for pywhispercpp (or set WHISPER_CPP_MODEL)",
    )
    parser.add_argument(
        "--whisper-cpp-models-dir",
        default=os.getenv("WHISPER_CPP_MODELS_DIR"),
        help="Optional model download/cache directory for pywhispercpp",
    )
    parser.add_argument(
        "--language", help="Language code for transcription (e.g., 'en', 'es')"
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=120,
        help="Maximum informative sentences retained for LLM input",
    )
    parser.add_argument(
        "--llm-model",
        default="deepseek-reasoner",
        help="Model name for note generation",
    )

    args = parser.parse_args()

    # Get API key from environment variables
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning(
            "DEEPSEEK_API_KEY not found in .env file. Typst generation will use template fallback."
        )

    try:
        processor = LectureProcessor(
            api_key=api_key,
            whisper_cpp_model=args.whisper_cpp_model,
            whisper_cpp_models_dir=args.whisper_cpp_models_dir,
            llm_model=args.llm_model,
        )
        typst_file = processor.process_lecture(
            args.video_path,
            args.output,
            args.keep_intermediates,
            args.language,
            args.max_sentences,
        )
        print(f"Success! Typst notes generated: {typst_file}")
    except Exception as e:
        logger.error(f"Error processing lecture: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
