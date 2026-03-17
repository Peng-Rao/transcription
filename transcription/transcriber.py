import logging
import os
from pathlib import Path

from pywhispercpp.model import Model

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, model_path="base.en", models_dir=None, **model_params):
        """
        Initialize whisper.cpp transcriber via pywhispercpp binding.

        Args:
            model_path (str): Model name (e.g. 'base.en') or local ggml model path
            models_dir (str | None): Optional model download directory
            **model_params: Additional pywhispercpp Model parameters
        """
        self.model_path = model_path
        self.models_dir = models_dir
        self.model_params = model_params
        self.model = self._create_model()

    def _create_model(self):
        """Create pywhispercpp model instance."""
        model_kwargs = dict(self.model_params)
        if self.models_dir:
            model_kwargs["models_dir"] = self.models_dir

        try:
            return Model(self.model_path, **model_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pywhispercpp model: {e}")

    def transcribe_to_srt(self, audio_path, srt_path, language=None):
        """
        Transcribe audio to SRT subtitle format using whisper.cpp

        Args:
            audio_path (str): Path to audio file
            srt_path (str): Path for output SRT file
            language (str): Language code (e.g., 'en', 'es'). None for auto-detect
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing audio file with pywhispercpp: {audio_path}")

        transcribe_params = {"print_realtime": False, "print_progress": False}
        if language:
            transcribe_params["language"] = language

        try:
            segments = self.model.transcribe(audio_path, **transcribe_params)
            self._write_srt_from_segments(segments, srt_path)
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio with pywhispercpp: {e}")

        logger.info(f"Transcription completed: {srt_path}")

    def _write_srt_from_segments(self, segments, srt_path):
        """Write pywhispercpp segments to SRT format."""
        srt_output = Path(srt_path)
        srt_output.parent.mkdir(parents=True, exist_ok=True)

        with open(srt_output, "w", encoding="utf-8") as f:
            index = 1
            for segment in segments:
                text = getattr(segment, "text", "").strip()
                if not text:
                    continue

                # pywhispercpp timestamps are in 10ms ticks.
                start_ms = int(getattr(segment, "t0", 0)) * 10
                end_ms = int(getattr(segment, "t1", 0)) * 10
                f.write(f"{index}\n")
                f.write(
                    f"{self._ms_to_srt_time(start_ms)} --> {self._ms_to_srt_time(end_ms)}\n"
                )
                f.write(f"{text}\n\n")
                index += 1

    def _ms_to_srt_time(self, milliseconds):
        """Convert milliseconds to SRT time format."""
        total_seconds, ms = divmod(milliseconds, 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

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
        return {
            "audio_file": audio_path,
            "model_path": str(self.model_path),
            "models_dir": str(self.models_dir) if self.models_dir else None,
            "language": "set during transcription",
            "backend": "pywhispercpp",
        }
