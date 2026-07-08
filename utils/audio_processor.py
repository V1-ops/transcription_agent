import os

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloades"
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TRANSCRIPTION_CHUNK_MINUTES = int(os.getenv("TRANSCRIPTION_CHUNK_MINUTES", "10"))

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _normalize_audio(audio: AudioSegment) -> AudioSegment:
    return audio.set_channels(TARGET_CHANNELS).set_frame_rate(TARGET_SAMPLE_RATE)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to a speech-friendly WAV."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    _normalize_audio(audio).export(output_path, format="wav")
    return output_path


def prepare_audio(source_path: str) -> str:
    """
    Normalize every source to mono 16kHz WAV so downstream upload slices stay
    small and consistent regardless of input format or source.
    """
    normalized_path = os.path.splitext(source_path)[0] + "_normalized.wav"
    audio = AudioSegment.from_file(source_path)
    _normalize_audio(audio).export(normalized_path, format="wav")
    return normalized_path


def chunk_audio(wav_path: str, chunk_minutes: int = TRANSCRIPTION_CHUNK_MINUTES) -> list[str]:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list[str]:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        source_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        source_path = convert_to_wav(source)

    print("Normalizing audio...")
    wav_path = prepare_audio(source_path)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
