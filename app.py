import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

def load_streamlit_secrets_into_env() -> None:
    """Mirror Streamlit Cloud secrets into environment variables for shared core modules."""
    try:
        secret_keys = [
            "MISTRAL_API_KEY",
            "WHISPER_MODEL",
            "SARVAM_API_KEY",
            "SARVAM_STT_MODEL",
        ]
        for key in secret_keys:
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        # Local runs may not have Streamlit secrets configured.
        pass

load_streamlit_secrets_into_env()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summary import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.Rag_pipeline import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(page_title="AI Meeting Assistant", page_icon="🎥", layout="wide")

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
# Streamlit reruns the ENTIRE script top-to-bottom on every interaction
# (button click, tab switch, chat message, etc). Without session_state,
# your pipeline results and chat history would vanish on every rerun.
if "result" not in st.session_state:
    st.session_state.result = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------------------------------------------------------
# Pipeline runner with live progress feedback
# ----------------------------------------------------------------------------
def run_pipeline(source: str, language: str) -> dict:
    status = st.empty()
    progress = st.progress(0)

    status.info("🎬 Processing audio/video input...")

    def on_audio_status(message: str):
        status.info(f"🎬 {message}")

    try:
        chunks = process_input(source, status_cb=on_audio_status)
    except TypeError:
        # Backward compatibility if an older audio_processor is imported.
        chunks = process_input(source)
    progress.progress(25)

    status.info("📝 Transcribing audio...")
    transcript = transcribe_all(chunks, language)
    progress.progress(55)

    status.info("🏷️ Generating title...")
    title = generate_title(transcript)
    progress.progress(65)

    status.info("📋 Generating summary...")
    summary = summarize(transcript)
    progress.progress(75)

    status.info("✅ Extracting action items...")
    action_items = extract_action_items(transcript)
    progress.progress(82)

    status.info("🔑 Extracting key decisions...")
    decisions = extract_key_decisions(transcript)
    progress.progress(88)

    status.info("❓ Extracting open questions...")
    questions = extract_questions(transcript)
    progress.progress(94)

    status.info("🔗 Building RAG index for chat...")
    rag_chain = build_rag_chain(transcript)
    progress.progress(100)

    status.empty()
    progress.empty()

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ----------------------------------------------------------------------------
# Sidebar — inputs and controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎥 AI Audio Assistant")
    st.caption("Turn a recording into a summary, action items, and a chat you can query.")

    st.divider()

    input_mode = st.radio("Input source", ["YouTube URL", "Local file path", "Upload a file"])

    source = None
    if input_mode == "YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    elif input_mode == "Local file path":
        source = st.text_input("File path", placeholder="C:\\path\\to\\video.mp4")
    else:
        uploaded = st.file_uploader("Upload audio/video", type=["mp4", "mp3", "wav", "m4a", "mov"])
        if uploaded is not None:
            # process_input() expects a path/source string, not an in-memory
            # UploadedFile object, so we persist it to a temp file on disk first.
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                source = tmp.name

    language = st.selectbox("Language", ["english", "hinglish"])

    run_clicked = st.button(
        "🚀 Process Audio",
        type="primary",
        use_container_width=True,
        disabled=not source,
    )

    if st.session_state.result:
        st.divider()
        if st.button("🔄 Start a new meeting", use_container_width=True):
            st.session_state.result = None
            st.session_state.messages = []
            st.rerun()


# ----------------------------------------------------------------------------
# Trigger pipeline
# ----------------------------------------------------------------------------
if run_clicked and source:
    try:
        st.session_state.result = run_pipeline(source, language)
        st.session_state.messages = []
    except Exception as e:
        st.error(f"Pipeline failed: {e}")

result = st.session_state.result

# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
if result is None:
    st.title("🎥 AI Audio Assistant")
    st.write(
        "Provide a YouTube URL, local file path, or upload a recording in the "
        "sidebar, then click **Process Meeting**."
    )
    st.info(
        "You'll get a title, summary, action items, key decisions, open "
        "questions, the full transcript, and a chat tab to ask questions "
        "about the meeting."
    )
else:
    st.title(f"📌 {result['title']}")

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "📄 Transcript", "💬 Chat"]
    )

    with tab_summary:
        st.markdown(result["summary"])

    with tab_actions:
        st.markdown(result["action_items"])

    with tab_decisions:
        st.markdown(result["key_decisions"])

    with tab_questions:
        st.markdown(result["open_questions"])

    with tab_transcript:
        st.text_area("Full transcript", result["transcript"], height=500)
        st.download_button(
            "⬇️ Download transcript",
            result["transcript"],
            file_name="transcript.txt",
        )

    with tab_chat:
        st.caption("Ask questions about this Audio — answers are grounded in the transcript via RAG.")

        # Replay prior turns so the chat log survives reruns
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask something about the audio...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_question(result["rag_chain"], question)
                    st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
