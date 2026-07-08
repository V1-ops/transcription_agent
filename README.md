# AI video Assistant

An end-to-end AI Audio/video intelligence app that converts long-form audio or video into structured meeting notes, action items, key decisions, open questions, and a transcript-grounded chat experience.

Built for real-world meeting workflows, this project accepts a YouTube link, a local media file, or a direct upload, then runs a multi-stage pipeline for audio extraction, transcription, summarization, information extraction, and retrieval-augmented question answering.

## Why This Project Matters

Teams spend a lot of time manually reviewing recordings, writing notes, and chasing follow-ups after meetings. This project automates that workflow by turning raw recordings into searchable and actionable outputs:

- Professional meeting title
- Bullet-point summary
- Action items with owner and deadline extraction
- Key decisions
- Open questions and follow-ups
- Full transcript
- RAG-based chat over the meeting content

## Key Features

- Multiple input modes
  - YouTube URL
  - Local file path
  - File upload through Streamlit
- Audio preprocessing pipeline
  - Downloads audio from YouTube using `yt-dlp`
  - Converts media to mono `16kHz` WAV using `pydub` and FFmpeg
  - Splits long recordings into deployment-friendly chunks
- Flexible transcription engine
  - Hugging Face Inference API for English transcription
  - Sarvam AI speech-to-text translation pipeline for Hinglish audio
- LLM-powered meeting intelligence
  - Meeting title generation
  - Chunked transcript summarization with map-reduce style processing
  - Action item, decision, and open-question extraction
- Retrieval-Augmented Generation
  - Embeds transcript chunks into ChromaDB
  - Supports grounded Q&A over the meeting transcript
- Two usage modes
  - Streamlit web app
  - Command-line interface

## Demo Workflow

1. Provide a YouTube URL, local file path, or upload a recording.
2. The app downloads or converts the media into WAV format.
3. Audio is chunked for scalable transcription.
4. The transcript is generated using Hugging Face ASR or Sarvam.
5. Mistral-powered chains produce a title, summary, action items, decisions, and open questions.
6. The transcript is indexed in ChromaDB.
7. Users can ask natural-language questions grounded in the meeting transcript.

## Architecture

```text
Input Source
  -> Audio Download / Conversion
  -> WAV Chunking
  -> Transcription
  -> Transcript
  -> LLM Summarization + Information Extraction
  -> Vector Embedding + ChromaDB
  -> RAG Chat Interface
```

## Tech Stack

- `Python`
- `Streamlit` for the frontend
- `yt-dlp` for YouTube audio extraction
- `pydub` + `FFmpeg` for audio conversion and chunking
- `huggingface_hub` `InferenceClient` for hosted speech-to-text
- `Sarvam AI` for Hinglish transcription + translation
- `LangChain` for orchestration
- `Mistral AI` for title generation, summarization, and extraction
- `sentence-transformers` for embeddings
- `ChromaDB` for local vector storage

## Project Structure

```text
video summarizer/
|-- app.py                    # Streamlit application
|-- main.py                   # CLI pipeline entry point
|-- test.py                   # Basic local test script
|-- requirements.txt
|-- core/
|   |-- extractor.py          # Action items, decisions, open questions
|   |-- Rag_pipeline.py       # RAG chain construction and chat querying
|   |-- summary.py            # Title generation and transcript summarization
|   |-- transcriber.py        # Hugging Face + Sarvam transcription routing
|   `-- vectore_store.py      # ChromaDB vector store utilities
|-- utils/
|   `-- audio_processor.py    # Downloading, conversion, chunking
|-- downloads/                # Generated media artifacts
`-- vector_db/                # Persisted Chroma vector database
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

This project depends on FFmpeg for media conversion.

- Windows: install FFmpeg and ensure it is available in `PATH`
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
HF_TOKEN=your_huggingface_token
HF_PROVIDER=hf-inference
HF_ASR_MODEL=
HF_TIMEOUT_SECONDS=300
TRANSCRIPTION_CHUNK_MINUTES=10
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_STT_MODEL=saaras:v2.5
```

## Running the App

### Streamlit UI

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, then:

- choose an input mode
- provide the media source
- select `english` or `hinglish`
- click `Process Meeting`

### CLI Mode

```bash
python main.py
```

The CLI will ask for:

- a YouTube URL or local file path
- transcription language

After processing, it prints the title, summary, action items, decisions, and open questions, then starts an interactive transcript-grounded chat loop.

## Supported Inputs

- `YouTube` video links
- `mp4`
- `mp3`
- `wav`
- `m4a`
- `mov`

## How the Intelligence Layer Works

### Summarization

The transcript is split into chunks using a recursive text splitter. Each chunk is summarized individually, then the chunk summaries are combined into a final professional meeting summary.

### Information Extraction

Separate prompt chains extract:

- action items
- key decisions
- unresolved questions or follow-ups

### RAG Chat

The transcript is split into smaller semantic chunks and embedded with `all-MiniLM-L6-v2`. These chunks are stored in `ChromaDB`, and relevant context is retrieved at query time so answers stay grounded in the actual meeting content.

## Example Use Cases

- Summarizing internal team standups
- Converting recorded client calls into action points
- Extracting follow-ups from product or engineering discussions
- Reviewing interview recordings
- Searching long meeting transcripts without rereading the full recording

## Resume-Ready Highlights

If you want to present this project on your resume, these are the strongest technical talking points:

- Built an end-to-end AI meeting assistant that transforms raw audio/video into structured summaries, action items, decisions, and transcript-grounded Q&A.
- Integrated multimodal preprocessing with YouTube ingestion, WAV conversion, and chunk-based transcription for long-form recordings.
- Implemented a hybrid transcription workflow using Hugging Face hosted ASR for English and Sarvam AI for Hinglish speech-to-text translation.
- Designed a retrieval-augmented generation pipeline using LangChain, sentence-transformer embeddings, and ChromaDB for contextual chat over meeting transcripts.
- Developed both a Streamlit web interface and a CLI workflow to support interactive and developer-friendly usage.

## Challenges Solved

- Handling long audio recordings by chunking both media and transcript content
- Supporting multilingual or mixed-language meeting scenarios
- Keeping Q&A grounded in transcript evidence instead of generic LLM responses
- Managing an end-to-end pipeline from raw media ingestion to final structured output

## Current Limitations

- The quality of summaries depends on transcription quality
- Speaker diarization is not currently enabled
- Very large recordings may take time because transcription and embedding are compute-heavy
- The current vector database persists locally and is not yet designed for multi-user deployment

## Future Improvements

- Add speaker diarization and speaker-attributed notes
- Export outputs as PDF, DOCX, or email-ready meeting minutes
- Add cloud deployment and user authentication
- Support meeting history and transcript search across sessions
- Add cost/performance controls for different LLM and embedding backends

## Notes for GitHub Presentation

To make this repository look strong on GitHub:

- add screenshots or a short demo GIF of the Streamlit interface
- include a sample transcript and sample output in a `samples/` folder
- add an `.env.example` file with placeholder values only
- avoid committing secrets or generated local database artifacts unless intentional

## License

You can add your preferred license here, for example `MIT`.

## Author

Replace this section with your name and links:

- Name: `Your Name`
- GitHub: `https://github.com/your-username`
- LinkedIn: `https://linkedin.com/in/your-profile`
