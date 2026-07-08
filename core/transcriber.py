import os

import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pydub import AudioSegment

load_dotenv()

SARVAM_PIECE_SECONDS = 25
HF_ASR_MODEL = os.getenv("HF_ASR_MODEL", "openai/whisper-large-v3").strip()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_TIMEOUT_SECONDS = int(os.getenv("HF_TIMEOUT_SECONDS", "300"))
HF_PROVIDER = os.getenv("HF_PROVIDER", "hf-inference")
HF_ASR_PIECE_SECONDS = int(os.getenv("HF_ASR_PIECE_SECONDS", "30"))
HF_ASR_MIN_PIECE_SECONDS = int(os.getenv("HF_ASR_MIN_PIECE_SECONDS", "10"))
HF_ASR_UPLOAD_FORMAT = os.getenv("HF_ASR_UPLOAD_FORMAT", "mp3")
HF_ASR_UPLOAD_BITRATE = os.getenv("HF_ASR_UPLOAD_BITRATE", "32k")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_hf_client = None


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClient(
            provider=HF_PROVIDER,
            token=HF_TOKEN,
            timeout=HF_TIMEOUT_SECONDS,
        )
    return _hf_client


def _is_request_too_large_error(error: Exception) -> bool:
    message = str(error).lower()
    return "413" in message or "request entity too large" in message or "request_too_large" in message


def _is_model_not_supported_error(error: Exception) -> bool:
    message = str(error).lower()
    return "model not supported by provider" in message or "not supported by provider" in message


def _transcribe_with_huggingface(
    audio: AudioSegment,
    chunk_path: str,
    piece_seconds: int,
    model_name: str | None,
) -> str:
    client = _get_hf_client()
    piece_ms = piece_seconds * 1000
    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        piece_path = f"{chunk_path}_hf_{piece_seconds}s_{i}.{HF_ASR_UPLOAD_FORMAT}"
        piece.export(piece_path, format=HF_ASR_UPLOAD_FORMAT, bitrate=HF_ASR_UPLOAD_BITRATE)

        try:
            piece_size_kb = os.path.getsize(piece_path) / 1024
            print(
                f"  -> HF piece {i + 1}/{total_pieces} "
                f"({piece_seconds}s, {piece_size_kb:.1f} KB)"
            )
            if model_name:
                result = client.automatic_speech_recognition(
                    piece_path,
                    model=model_name,
                )
            else:
                result = client.automatic_speech_recognition(piece_path)
            full_text += result.text.strip() + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk_huggingface(chunk_path: str) -> str:
    audio = AudioSegment.from_wav(chunk_path)
    audio = audio.set_channels(1).set_frame_rate(16000)

    candidate_models = [HF_ASR_MODEL, None] if HF_ASR_MODEL else [None]

    for index, model_name in enumerate(candidate_models):
        piece_seconds = HF_ASR_PIECE_SECONDS
        while piece_seconds >= HF_ASR_MIN_PIECE_SECONDS:
            try:
                return _transcribe_with_huggingface(audio, chunk_path, piece_seconds, model_name)
            except Exception as error:
                if _is_model_not_supported_error(error) and model_name and index < len(candidate_models) - 1:
                    print(
                        f"  -> Hugging Face provider '{HF_PROVIDER}' does not support "
                        f"model '{model_name}'. Retrying with the provider default ASR model."
                    )
                    break

                if _is_request_too_large_error(error) and piece_seconds > HF_ASR_MIN_PIECE_SECONDS:
                    next_piece_seconds = max(HF_ASR_MIN_PIECE_SECONDS, piece_seconds // 2)
                    if next_piece_seconds == piece_seconds:
                        raise
                    print(
                        f"  -> Hugging Face returned 413 at {piece_seconds}s pieces. "
                        f"Retrying with {next_piece_seconds}s pieces."
                    )
                    piece_seconds = next_piece_seconds
                    continue
                raise

    raise RuntimeError("Hugging Face transcription failed after exhausting fallback piece sizes.")


def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\nSarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts <=30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  -> Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Hugging Face or Sarvam depending on language choice.
    - english  -> Hugging Face Inference API
    - hinglish -> Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_huggingface(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""

    engine = "Sarvam AI" if language.lower() == "hinglish" else f"Hugging Face ASR ({HF_ASR_MODEL})"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription complete.")
    return full_transcript.strip()
