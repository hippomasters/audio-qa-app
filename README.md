# Audio QA Pipeline with Local LLM

This project is a complete pipeline that takes an audio file, performs speaker diarization, transcribes the speech, and creates a searchable knowledge base. You can then ask questions about the audio content and receive cited answers from a locally running LLM.

## Core Scripts
- `create_timeline.py`
- `transcribe_hf.py`
- `index_transcript.py`
- `ask_question.py`

## How to Run
1.  **Setup:** Install dependencies from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Pipeline:**
    ```bash
    # Clear old data (recommended for new audio)
    rm -rf ./chroma_db

    # Run the processing steps
    python create_timeline.py
    python transcribe_hf.py
    python index_transcript.py

    # Start the QA app
    python ask_question.py
    ```
