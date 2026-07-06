import yt_dlp
import os


def _get_audio_segment():
    from pydub import AudioSegment

    return AudioSegment


DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# converting a youtube video into a wavfile
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
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    _status("Connecting to YouTube...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        original_name = ydl.prepare_filename(info)

    wav_path = os.path.splitext(original_name)[0] + ".wav"
    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"YouTube download finished but WAV file was not found: {wav_path}. "
            "Check FFmpeg installation and yt-dlp postprocessing output."
        )
    _status("YouTube audio prepared.")
    return wav_path

# converting any audio/video file to wav format 

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    AudioSegment = _get_audio_segment()
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16kHz
    audio.export(output_path, format="wav")
    return output_path
# chunking the audio

def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    AudioSegment = _get_audio_segment()
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks
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
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    _status("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    _status(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=0OpImRKSYPc"
    final_chunks = process_input(url)
    print(final_chunks)

