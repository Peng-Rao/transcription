import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from transcription.audio_extractor import AudioExtractor
from transcription.latex_generator import LaTeXGenerator
from transcription.text_processor import TextProcessor
from transcription.transcriber import Transcriber

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LectureProcessor:
    def __init__(self, api_key=None, whisper_model="base"):
        self.audio_extractor = AudioExtractor()
        self.transcriber = Transcriber(model_size=whisper_model)
        self.text_processor = TextProcessor()
        self.latex_generator = LaTeXGenerator(
            api_key=api_key, model="deepseek-reasoner"
        )

    def process_lecture(
        self, video_path, output_dir="output", keep_intermediates=False, language=None
    ):
        """
        Complete pipeline to process a lecture video into LaTeX notes

        Args:
            video_path (str): Path to the input video file
            output_dir (str): Directory to save output files
            keep_intermediates (bool): Whether to keep intermediate files
            language (str): Language code for transcription (e.g., 'en', 'es')

        Returns:
            str: Path to the generated LaTeX file
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Processing lecture: {video_path.name}")

        # Step 1: Extract audio
        logger.info("Step 1: Extracting audio from video...")
        audio_path = output_dir / f"{video_path.stem}_audio.wav"
        self.audio_extractor.extract_audio(str(video_path), str(audio_path))

        # Step 2: Transcribe audio to subtitles
        logger.info("Step 2: Transcribing audio to subtitles using Whisper...")
        srt_path = output_dir / f"{video_path.stem}_subtitles.srt"
        self.transcriber.transcribe_to_srt(
            str(audio_path), str(srt_path), language=language
        )

        # Step 3: Extract and process text
        logger.info("Step 3: Processing extracted text...")
        raw_text = self.transcriber.extract_text_from_srt(str(srt_path))
        processed_text = self.text_processor.process_text(raw_text)

        # Save processed text
        text_path = output_dir / f"{video_path.stem}_processed.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(processed_text)

        # Step 4: Generate LaTeX notes
        logger.info("Step 4: Generating LaTeX notes...")
        latex_path = output_dir / f"{video_path.stem}_notes.tex"
        self.latex_generator.generate_notes(
            processed_text, str(latex_path), title=f"Lecture Notes: {video_path.stem}"
        )

        # Cleanup intermediate files if requested
        if not keep_intermediates:
            if audio_path.exists():
                audio_path.unlink()
            if srt_path.exists():
                srt_path.unlink()

        logger.info(f"Processing complete! LaTeX notes saved to: {latex_path}")
        return str(latex_path)


def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Convert lecture videos to LaTeX notes"
    )
    parser.add_argument("video_path", help="Path to the lecture video file")
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
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--language", help="Language code for transcription (e.g., 'en', 'es')"
    )

    args = parser.parse_args()

    # Get API key from environment variables
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning(
            "DEEPSEEK_API_KEY not found in .env file. LaTeX generation will use template fallback."
        )

    try:
        processor = LectureProcessor(
            api_key=api_key,  # Pass API key from environment
            whisper_model=args.whisper_model,
        )
        latex_file = processor.process_lecture(
            args.video_path, args.output, args.keep_intermediates, args.language
        )
        print(f"Success! LaTeX notes generated: {latex_file}")
    except Exception as e:
        logger.error(f"Error processing lecture: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
