import argparse
import json
import logging
import os

import whisper

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, model_size="small"):
        """
        Initialize Whisper transcriber

        Args:
            model_size (str): Whisper model size - tiny, base, small, medium, large
                             Larger models are more accurate but slower
        """
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        self.model_size = model_size

    def transcribe_to_srt(self, audio_path, srt_path, language=None):
        """
        Transcribe audio to SRT subtitle format using Whisper

        Args:
            audio_path (str): Path to audio file
            srt_path (str): Path for output SRT file
            language (str): Language code (e.g., 'en', 'es'). None for auto-detect
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing audio file: {audio_path}")

        # Transcribe with Whisper
        transcribe_options = {"task": "transcribe", "verbose": False}

        if language:
            transcribe_options["language"] = language

        try:
            result = self.model.transcribe(audio_path, **transcribe_options)

            # Extract segments with timestamps
            segments = result.get("segments", [])

            if not segments:
                logger.warning("No speech segments detected in audio")
                # Create empty SRT file
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write("")
                return

            # Write SRT file
            self._write_srt_from_segments(segments, srt_path)

            logger.info(f"Transcription completed: {srt_path}")
            logger.info(f"Detected language: {result.get('language', 'unknown')}")
            logger.info(f"Total segments: {len(segments)}")

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}")

    def transcribe_with_word_timestamps(self, audio_path, output_path=None):
        """
        Transcribe with word-level timestamps (requires newer Whisper versions)

        Args:
            audio_path (str): Path to audio file
            output_path (str): Optional path to save detailed JSON output

        Returns:
            dict: Detailed transcription result with word timestamps
        """
        logger.info(f"Transcribing with word timestamps: {audio_path}")

        try:
            result = self.model.transcribe(
                audio_path, task="transcribe", word_timestamps=True, verbose=False
            )

            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Detailed transcription saved: {output_path}")

            return result

        except Exception as e:
            logger.error(f"Word-level transcription error: {e}")
            # Fallback to regular transcription
            return self.model.transcribe(audio_path, task="transcribe", verbose=False)

    def _write_srt_from_segments(self, segments, srt_path):
        """Write Whisper segments to SRT format"""
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._seconds_to_srt_time(segment["start"])
                end_time = self._seconds_to_srt_time(segment["end"])
                text = segment["text"].strip()

                # Skip empty segments
                if not text:
                    continue

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")

    def _seconds_to_srt_time(self, seconds):
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def extract_text_from_srt(self, srt_path):
        """Extract plain text from SRT file"""
        if not os.path.exists(srt_path):
            raise FileNotFoundError(f"SRT file not found: {srt_path}")

        text_lines = []

        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            logger.warning("SRT file is empty")
            return ""

        # Split by double newlines to get subtitle blocks
        blocks = content.strip().split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:  # Valid subtitle block
                # Skip sequence number and timestamp, get text
                text = "\n".join(lines[2:])
                text_lines.append(text)

        return " ".join(text_lines)

    def get_transcription_info(self, audio_path):
        """
        Get information about the transcription without full processing

        Args:
            audio_path (str): Path to audio file

        Returns:
            dict: Basic info about the audio and expected transcription
        """
        try:
            # Quick transcription to get language and basic info
            result = self.model.transcribe(audio_path, task="transcribe", verbose=False)

            return {
                "language": result.get("language", "unknown"),
                "duration": len(result.get("segments", [])),
                "text_preview": result.get("text", "")[:200] + "..."
                if len(result.get("text", "")) > 200
                else result.get("text", ""),
                "model_used": self.model_size,
            }

        except Exception as e:
            logger.error(f"Error getting transcription info: {e}")
            return {
                "language": "unknown",
                "duration": 0,
                "text_preview": "",
                "model_used": self.model_size,
                "error": str(e),
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Whisper Audio Transcriber")
    parser.add_argument("audio_path", help="Path to the input audio file")
    parser.add_argument(
        "--srt", dest="srt_path", default="output.srt", help="Path to output SRT file"
    )
    parser.add_argument(
        "--language", default=None, help="Optional language code (e.g., 'en', 'zh')"
    )

    args = parser.parse_args()

    transcriber = Transcriber(model_size="small")

    try:
        # Transcribe to SRT
        transcriber.transcribe_to_srt(
            audio_path=args.audio_path, srt_path=args.srt_path, language=args.language
        )

        # Display extracted plain text
        plain_text = transcriber.extract_text_from_srt(args.srt_path)
        print("\n--- Extracted Text ---\n")
        print(plain_text)

        # Display basic info
        info = transcriber.get_transcription_info(args.audio_path)
        print("\n--- Transcription Info ---")
        for k, v in info.items():
            print(f"{k}: {v}")

    except Exception as e:
        logger.error(f"Failed to process audio: {e}")
