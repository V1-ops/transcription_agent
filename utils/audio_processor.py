import os
import subprocess

import yt_dlp


DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg is required but was not found in PATH. Install FFmpeg and try again."
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "Unknown FFmpeg error"
        raise RuntimeError(message) from exc


def _probe_duration_seconds(path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFprobe is required but was not found in PATH. Install FFmpeg and try again."
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "Unknown FFprobe error"
        raise RuntimeError(message) from exc

    return float(result.stdout.strip())


def _export_wav_segment(input_path: str, output_path: str, start_seconds: float, duration_seconds: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-t",
        str(duration_seconds),
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        output_path,
    ]
    _run_command(command)


def split_wav_file(wav_path: str, chunk_seconds: int, suffix_prefix: str = "_chunk_") -> list[str]:
    duration_seconds = _probe_duration_seconds(wav_path)
    if duration_seconds <= 0:
        raise RuntimeError(f"Could not determine audio duration for {wav_path}")

    chunks = []
    start_seconds = 0.0
    index = 0

    while start_seconds < duration_seconds:
        output_path = f"{wav_path}{suffix_prefix}{index}.wav"
        segment_duration = min(chunk_seconds, max(duration_seconds - start_seconds, 0))
        _export_wav_segment(wav_path, output_path, start_seconds, segment_duration)
        chunks.append(output_path)
        start_seconds += chunk_seconds
        index += 1

    return chunks


def normalize_to_wav(input_path: str) -> str:
    """Convert any audio/video file to mono 16kHz WAV using FFmpeg."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        output_path,
    ]
    _run_command(command)
    return output_path


def download_youtube_audio(url: str, status_cb=None) -> str:
    def _status(message: str):
        if status_cb:
            status_cb(message)

    def _progress_hook(d):
        state = d.get("status")
        if state == "downloading":
            downloaded = d.get("downloaded_bytes")
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if downloaded and total:
                pct = (downloaded / total) * 100
                _status(f"Downloading from YouTube... {pct:.1f}%")
            else:
                _status("Downloading from YouTube...")
        elif state == "finished":
            _status("Download complete. Converting to WAV...")

    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "socket_timeout": 25,
        "retries": 3,
        "fragment_retries": 3,
        "progress_hooks": [_progress_hook],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    _status("Connecting to YouTube...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        original_path = ydl.prepare_filename(info)

    if not os.path.exists(original_path):
        raise FileNotFoundError(f"YouTube download finished but file was not found: {original_path}")

    wav_path = normalize_to_wav(original_path)
    _status("YouTube audio prepared.")
    return wav_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list[str]:
    return split_wav_file(wav_path, chunk_seconds=chunk_minutes * 60, suffix_prefix="_chunk_")


def process_input(source: str, status_cb=None) -> list:
    def _status(message: str):
        if status_cb:
            status_cb(message)

    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        _status("Detected YouTube URL.")
        wav_path = download_youtube_audio(source, status_cb=_status)
    else:
        print("Detected local file. Converting to WAV...")
        _status("Detected local file. Converting to WAV...")
        wav_path = normalize_to_wav(source)

    print("Chunking audio...")
    _status("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    _status(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=0OpImRKSYPc"
    final_chunks = process_input(url)
    print(final_chunks)
