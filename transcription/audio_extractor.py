import logging
import os
import subprocess

logger = logging.getLogger(__name__)


class AudioExtractor:
    def __init__(self):
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Check if ffmpeg is available"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "ffmpeg is not installed or not in PATH. "
                "Please install ffmpeg to use this tool."
            )

    def extract_audio(self, video_path, audio_path, sample_rate=16000):
        """
        Extract audio from video using ffmpeg

        Args:
            video_path (str): Path to input video file
            audio_path (str): Path for output audio file
            sample_rate (int): Audio sample rate (default: 16000 for speech)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # ffmpeg command to extract audio
        cmd = [
            "ffmpeg",
            "-i",
            video_path,  # Input video
            "-vn",  # Disable video
            "-acodec",
            "pcm_s16le",  # Audio codec
            "-ar",
            str(sample_rate),  # Sample rate
            "-ac",
            "1",  # Mono channel
            "-y",  # Overwrite output file
            audio_path,
        ]

        logger.info(f"Extracting audio: {video_path} -> {audio_path}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Audio extraction completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to extract audio: {e.stderr}")
