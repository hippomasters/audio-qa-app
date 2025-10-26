import torch
import torchaudio
from transformers import pipeline
from pyannote.audio import Pipeline
import chromadb
from sentence_transformers import SentenceTransformer
from collections import Counter
import os
import json
import ffmpeg
import uuid
import ollama
import streamlit as st

# --- Engine Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL_ID = "openai/whisper-base"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
OLLAMA_MODEL = "llama3:8b" # Define the model name here
# --- Chunking Configuration ---
TARGET_CHUNK_WORD_COUNT = 250
MIN_CHUNK_WORD_COUNT = 100

# --- Timeline Cleaning Configuration ---
MAX_PAUSE_S = 2.0

# --- Cached Model Loaders ---

@st.cache_resource
def load_diarization_pipeline():
    print("Cache Miss: Loading diarization pipeline for the first time.")
    try:
        # --- ADDED: Read token from secrets ---
        hf_token = st.secrets.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            st.error("Hugging Face Hub token not found in secrets.toml. Diarization will likely fail.")
            # Optionally raise an error or return None depending on desired handling

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token # Pass the token here
        )
        # ------------------------------------
        pipeline.to(torch.device(DEVICE))
        return pipeline
    except Exception as e:
        st.error(f"Failed to load diarization pipeline: {e}")
        return None # Return None on failure

@st.cache_resource
def load_whisper_pipeline():
    print("Cache Miss: Loading Whisper pipeline for the first time.")
    return pipeline("automatic-speech-recognition", model=WHISPER_MODEL_ID, device=DEVICE)

@st.cache_resource
def load_embedding_model():
    print("Cache Miss: Loading embedding model for the first time.")
    return SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

# --- Helper & Core Functions ---
def prepare_audio_from_upload(uploaded_file_path: str, session_folder: str):
    """
    Checks an uploaded file, extracts audio if it's a video using ffmpeg-python,
    and returns a path to a processable audio file.
    """
    filename = os.path.basename(uploaded_file_path)
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext in video_extensions:
        print(f"▶️  Engine: Video file detected. Extracting audio from '{filename}'...")
        try:
            audio_filename = f"extracted_audio_{uuid.uuid4()}.wav"
            output_audio_path = os.path.join(session_folder, audio_filename)
            ffmpeg.input(uploaded_file_path).output(
                output_audio_path,
                acodec='pcm_s16le', ac=1, ar='16k'
            ).run(quiet=True, overwrite_output=True)
            print(f"✅ Engine: Audio extracted successfully to '{output_audio_path}'")
            return output_audio_path
        except ffmpeg.Error as e:
            print(f"❌ Engine Error: Failed to extract audio from video. FFmpeg error:")
            print(e.stderr.decode())
            return None
    else:
        print(f"▶️  Engine: Audio file detected. Using '{filename}' directly.")
        return uploaded_file_path

def _clean_timeline(timeline: list):
    """
    Merges consecutive segments from the same speaker if the gap between them is short.
    """
    if not timeline:
        return []
    cleaned_timeline, current_chunk = [], timeline[0].copy()
    for i in range(1, len(timeline)):
        next_segment = timeline[i]
        gap = next_segment['start'] - current_chunk['end']
        if next_segment['speaker'] == current_chunk['speaker'] and gap < MAX_PAUSE_S:
            current_chunk['end'] = next_segment['end']
        else:
            cleaned_timeline.append(current_chunk)
            current_chunk = next_segment.copy()
    cleaned_timeline.append(current_chunk)
    return cleaned_timeline

def run_diarization(audio_path: str):
    """
    Runs speaker diarization on an audio file and returns the timeline data.
    """
    pipeline = load_diarization_pipeline()
    print(f"▶️  Engine: Analyzing '{os.path.basename(audio_path)}' for speakers...")
    try:
        diarization = pipeline(audio_path)
        timeline = [{"speaker": speaker, "start": round(turn.start, 2), "end": round(turn.end, 2)} for turn, _, speaker in diarization.itertracks(yield_label=True)]
        print("✅ Engine: Diarization complete.")
        return timeline
    except Exception as e:
        print(f"❌ Engine Error: Diarization failed for {audio_path}. {e}")
        return None

def run_transcription(audio_path: str, cleaned_timeline: list, session_folder: str, source_filename: str):
    """
    Transcribes audio segments by grouping them into larger chunks first.
    """
    transcribe_pipeline = load_whisper_pipeline()
    print(f"▶️  Engine: Loading and resampling audio file: '{os.path.basename(audio_path)}'")
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    except Exception as e:
        print(f"❌ Engine Error: Could not load or resample audio file {audio_path}. {e}")
        return None

    grouped_segments = []
    current_group = []
    for i, segment in enumerate(cleaned_timeline):
        current_group.append(segment)
        if i == len(cleaned_timeline) - 1 or (cleaned_timeline[i+1]['start'] - segment['end'] > MAX_PAUSE_S):
            grouped_segments.append(current_group)
            current_group = []

    full_transcript = []
    print(f"▶️  Engine: Transcribing {len(grouped_segments)} larger audio chunks...")
    for group in grouped_segments:
        start_time = group[0]['start']
        end_time = group[-1]['end']
        start_frame = int(start_time * 16000)
        end_frame = int(end_time * 16000)
        group_waveform = waveform[:, start_frame:end_frame]
        audio_for_hf = group_waveform.squeeze().numpy()
        result = transcribe_pipeline(audio_for_hf, return_timestamps=True)

        # **CORRECTION:** Creates a single, correct transcript object per group
        if result["text"].strip():
            full_transcript.append({
                "speaker": group[0]['speaker'],
                "start": start_time,
                "end": end_time,
                "text": result["text"].strip(),
                "source_file": source_filename
            })
    return full_transcript

def _determine_primary_speaker(chunk_buffer):
    if not chunk_buffer: return "Unknown"
    word_counts = Counter()
    for segment in chunk_buffer:
        speaker = segment.get("speaker", "Unknown")
        word_count = len(segment.get("text", "").split())
        word_counts[speaker] += word_count
    return word_counts.most_common(1)[0][0]

def _finalize_chunk(chunk_buffer, source_file):
    if not chunk_buffer: return None
    return {
        "primary_speaker": _determine_primary_speaker(chunk_buffer),
        "text": " ".join([seg["text"] for seg in chunk_buffer]),
        "start": chunk_buffer[0]["start"],
        "end": chunk_buffer[-1]["end"],
        "original_segments": json.dumps(chunk_buffer),
        "source_file": source_file
    }

def run_indexing(all_segments: list, session_folder: str):
    """
    Chunks a transcript, creates embeddings, and stores them in a session-specific ChromaDB.
    """
    model = load_embedding_model()
    db_path = os.path.join(session_folder, "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="audio_transcript")

    segments_by_file = {}
    for seg in all_segments:
        if not isinstance(seg, dict):
            print(f"⚠️  Engine Warning: Skipping malformed segment: {seg}")
            continue
        source = seg.get("source_file", "Unknown File")
        if source not in segments_by_file:
            segments_by_file[source] = []
        segments_by_file[source].append(seg)

    all_merged_chunks = []
    for source_file, segments in segments_by_file.items():
        print(f"▶️  Engine: Chunking {len(segments)} segments from '{source_file}'...")
        chunk_buffer = []
        for i, segment in enumerate(segments):
            chunk_buffer.append(segment)
            current_words = len(" ".join([s["text"] for s in chunk_buffer]).split())
            if current_words >= TARGET_CHUNK_WORD_COUNT or i == len(segments) - 1:
                # **CORRECTION:** Passes the source_file argument correctly
                final_chunk = _finalize_chunk(chunk_buffer, source_file)
                if final_chunk:
                    all_merged_chunks.append(final_chunk)
                chunk_buffer = []
        if chunk_buffer:
             # **CORRECTION:** Passes the source_file argument correctly for leftovers
             final_chunk = _finalize_chunk(chunk_buffer, source_file)
             if final_chunk:
                all_merged_chunks.append(final_chunk)

    print(f"✅ Engine: Merged all segments into {len(all_merged_chunks)} coherent chunks.")
    if not all_merged_chunks:
        print("No chunks to index.")
        return True

    ids = [str(uuid.uuid4()) for _ in all_merged_chunks]
    documents = [chunk["text"] for chunk in all_merged_chunks]
    metadatas = all_merged_chunks
    embeddings = model.encode(documents, show_progress_bar=True)
    collection.upsert(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=documents)
    print(f"✅ Engine: Successfully indexed. Collection now contains {collection.count()} documents.")
    return True


def process_session_pipeline(list_of_file_paths: list, session_folder: str):
    """
    The main processing pipeline that runs all steps and yields progress updates.
    """
    all_transcribed_segments = []
    total_files = len(list_of_file_paths)
    for i, file_path in enumerate(list_of_file_paths):
        original_filename = os.path.basename(file_path)
        yield f"({i+1}/{total_files}) Processing '{original_filename}'..."
        yield f"  - Preparing audio..."
        audio_path = prepare_audio_from_upload(file_path, session_folder)
        if not audio_path:
            yield f"ERROR: Failed to prepare audio for '{original_filename}'."
            continue

        yield f"  - Identifying speakers..."
        raw_timeline = run_diarization(audio_path)
        if not raw_timeline:
            yield f"ERROR: Diarization failed for '{original_filename}'."
            continue

        yield f"  - Cleaning up timeline..."
        cleaned_timeline = _clean_timeline(raw_timeline)

        yield f"  - Transcribing audio..."
        transcribed_segments = run_transcription(audio_path, cleaned_timeline, session_folder, original_filename)
        # --- ADDED: Clean up extracted audio file ---
        if audio_path != file_path: # Check if a temporary file was created
            try:
                os.remove(audio_path)
                print(f"✅ Engine: Cleaned up temporary audio file: {os.path.basename(audio_path)}")
            except OSError as e:
                print(f"⚠️ Engine Warning: Could not delete temporary audio file {audio_path}. Error: {e}")
        # --------------------------------------------
        if not transcribed_segments:
            yield f"ERROR: Transcription failed for '{original_filename}'."
            continue
        all_transcribed_segments.extend(transcribed_segments)

    if all_transcribed_segments:
        yield f"Indexing all transcripts for Q&A..."
        success = run_indexing(all_transcribed_segments, session_folder)
        if not success:
            yield "ERROR: Indexing failed."
            return
    else:
        yield "Warning: No files were successfully transcribed."
    yield "✅ All files processed! You can now ask questions."

# **CORRECTION:** Fixes the stray 'def' keyword for a valid syntax
def ask_question(question: str, session_folder: str):
    """
    Answers a question based on the indexed transcript for a session.
    """
    print(f"▶️  Engine: Answering question: '{question}'")
    embedding_model = load_embedding_model()
    db_path = os.path.join(session_folder, "chroma_db")
    if not os.path.exists(db_path):
        yield "ERROR: The database for this session has not been indexed yet."
        return

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="audio_transcript")
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        include=['metadatas', 'documents', 'distances']
    )
    if not results['documents'] or results['distances'][0][0] > 0.6:
        yield "I could not find a relevant answer in the provided transcript to answer that question."
        return

    context = ""
    cited_sources = set()
    for i, doc in enumerate(results['documents'][0]):
        metadata = results['metadatas'][0][i]
        speaker = metadata.get('primary_speaker', 'Unknown')
        source_file = metadata.get('source_file', 'Unknown File')
        start = metadata.get('start')
        end = metadata.get('end')
        context += f"Source {i+1} (Speaker: {speaker} from file '{source_file}' from {start:.2f}s to {end:.2f}s):\n"
        context += f"\"{doc}\"\n---\n"
        cited_sources.add((speaker, source_file, start, end))

    prompt_template = """
    You are an expert Q&A assistant. Your task is to synthesize a clear and concise answer from the provided context.
    Do not just copy-paste sentences from the sources. Read all the provided information and formulate a new, coherent response in your own words.
    Your response MUST be based ONLY on the provided context. If the answer is not in the context, state that clearly.
    DO NOT include any citations or source numbers in your answer.
    ---
    CONTEXT:
    {context}
    ---
    USER QUESTION:
    {question}
    """
    final_prompt = prompt_template.format(context=context, question=question)

    response_stream = ollama.chat(
    model=OLLAMA_MODEL, # Use the variable here
    messages=[{'role': 'user', 'content': final_prompt}],
    stream=True
)
    for chunk in response_stream:
        yield chunk['message']['content']

    if cited_sources:
        yield "\n\n**Sources:**\n"
        sorted_sources = sorted(list(cited_sources), key=lambda x: x[2])
        for speaker, source, start, end in sorted_sources:
            yield f"- *{speaker} from **{source}**: {start:.2f}s to {end:.2f}s*\n"
