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

# --- Engine Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL_ID = "openai/whisper-base"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# --- Chunking Configuration ---
TARGET_CHUNK_WORD_COUNT = 250
MIN_CHUNK_WORD_COUNT = 100

# --- NEW: Timeline Cleaning Configuration ---
# The maximum pause in seconds between segments from the same speaker to be merged
MAX_PAUSE_S = 2.0
#---Check if its a video file ---
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

            # Use ffmpeg-python to perform the audio extraction
            ffmpeg.input(uploaded_file_path).output(
                output_audio_path,
                acodec='pcm_s16le', # Standard codec for WAV
                ac=1, # Mono channel
                ar='16k' # 16kHz sample rate
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

    cleaned_timeline = []
    # Start with the first segment as the current chunk
    current_chunk = timeline[0].copy()

    for i in range(1, len(timeline)):
        next_segment = timeline[i]

        # Calculate the gap between the current chunk and the next segment
        gap = next_segment['start'] - current_chunk['end']

        # Check if the speaker is the same and the gap is short
        if next_segment['speaker'] == current_chunk['speaker'] and gap < MAX_PAUSE_S:
            # If so, merge by extending the end time of the current chunk
            current_chunk['end'] = next_segment['end']
        else:
            # If not, the current chunk is finalized. Add it to our list.
            cleaned_timeline.append(current_chunk)
            # The next segment becomes the new current chunk
            current_chunk = next_segment.copy()

    # Add the very last chunk to the list
    cleaned_timeline.append(current_chunk)

    return cleaned_timeline
# --- Diarization Function ---
def run_diarization(audio_path: str, session_folder: str):
    """
    Runs speaker diarization on an audio file and saves the timeline.
    """
    print("▶️  Engine: Loading diarization pipeline...")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        pipeline.to(torch.device(DEVICE))
        print("✅ Engine: Diarization pipeline loaded.")
    except Exception as e:
        print(f"❌ Engine Error: Could not load diarization pipeline. {e}")
        return None

    print(f"▶️  Engine: Analyzing '{os.path.basename(audio_path)}' for speakers...")
    try:
        diarization = pipeline(audio_path)
        print("✅ Engine: Diarization complete.")
    except Exception as e:
        print(f"❌ Engine Error: Diarization failed for {audio_path}. {e}")
        return None

    timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        timeline.append({"speaker": speaker, "start": round(turn.start, 2), "end": round(turn.end, 2)})

    output_path = os.path.join(session_folder, "timeline.json")
    with open(output_path, "w") as f:
        json.dump(timeline, f, indent=2)
        
    print(f"✅ Engine: Speaker timeline saved to '{output_path}'")
    return output_path

# --- Transcription Function ---
def run_transcription(audio_path: str, timeline_path: str, session_folder: str):
    """
    Transcribes audio segments based on a speaker timeline.
    """
    print(f"▶️  Engine: Loading Whisper pipeline for model: '{WHISPER_MODEL_ID}'...")
    try:
        transcribe_pipeline = pipeline("automatic-speech-recognition", model=WHISPER_MODEL_ID, device=DEVICE)
        print("✅ Engine: Whisper pipeline loaded.")
    except Exception as e:
        print(f"❌ Engine Error: Could not load Whisper pipeline. {e}")
        return None

    print(f"▶️  Engine: Loading and resampling audio file: '{os.path.basename(audio_path)}'")
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        print("✅ Engine: Audio loaded.")
    except Exception as e:
        print(f"❌ Engine Error: Could not load or resample audio file {audio_path}. {e}")
        return None

    with open(timeline_path, "r") as f:
        timeline = json.load(f)

    full_transcript = []
    print(f"▶️  Engine: Transcribing {len(timeline)} segments...")
    
    for segment in timeline:
        start_frame = int(segment["start"] * 16000)
        end_frame = int(segment["end"] * 16000)
        segment_waveform = waveform[:, start_frame:end_frame]
        
        audio_segment_for_hf = segment_waveform.squeeze().numpy()
        result = transcribe_pipeline(audio_segment_for_hf, return_timestamps=True)
        
        full_transcript.append({"speaker": segment["speaker"], "start": segment["start"], "end": segment["end"], "text": result["text"].strip()})
    
    output_path = os.path.join(session_folder, "final_transcript_hf.json")
    with open(output_path, "w") as f:
        json.dump(full_transcript, f, indent=2)

    print(f"✅ Engine: Transcription complete. Saved to '{output_path}'")
    return output_path

# --- Indexing Functions (from your index_transcript.py) ---
def _determine_primary_speaker(chunk_buffer):
    if not chunk_buffer: return "Unknown"
    word_counts = Counter(seg.get("speaker", "Unknown") for seg in chunk_buffer for _ in seg.get("text", "").split())
    return word_counts.most_common(1)[0][0]

def _finalize_chunk(chunk_buffer):
    if not chunk_buffer: return None
    return {
        "primary_speaker": _determine_primary_speaker(chunk_buffer),
        "text": " ".join([seg["text"] for seg in chunk_buffer]),
        "start": chunk_buffer[0]["start"],
        "end": chunk_buffer[-1]["end"],
        "original_segments": json.dumps(chunk_buffer)
    }

def run_indexing(transcript_path: str, session_folder: str):
    """
    Chunks a transcript, creates embeddings, and stores them in a session-specific ChromaDB.
    """
    print(f"▶️  Engine: Loading embedding model: '{EMBEDDING_MODEL}'...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Engine: Embedding model loaded.")
    except Exception as e:
        print(f"❌ Engine Error: Could not load embedding model. {e}")
        return False

    db_path = os.path.join(session_folder, "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="audio_transcript")
    print(f"✅ Engine: ChromaDB collection ready at '{db_path}'.")

    with open(transcript_path, "r") as f:
        transcript_data = json.load(f)
    print(f"▶️  Engine: Loaded {len(transcript_data)} raw segments.")

    merged_chunks, chunk_buffer = [], []
    for i, segment in enumerate(transcript_data):
        chunk_buffer.append(segment)
        current_words = len(" ".join([s["text"] for s in chunk_buffer]).split())
        if current_words >= TARGET_CHUNK_WORD_COUNT or i == len(transcript_data) - 1:
            merged_chunks.append(_finalize_chunk(chunk_buffer))
            chunk_buffer = []

    if chunk_buffer:
        if merged_chunks and len(" ".join([s['text'] for s in chunk_buffer]).split()) < MIN_CHUNK_WORD_COUNT:
            # Logic to merge small leftover chunk with the previous one
            last_main_chunk, last_buffer_chunk = merged_chunks[-1], _finalize_chunk(chunk_buffer)
            last_main_chunk['text'] += " " + last_buffer_chunk['text']
            last_main_chunk['end'] = last_buffer_chunk['end']
            original_segments = json.loads(last_main_chunk['original_segments'])
            original_segments.extend(json.loads(last_buffer_chunk['original_segments']))
            last_main_chunk['original_segments'] = json.dumps(original_segments)
        else:
            merged_chunks.append(_finalize_chunk(chunk_buffer))

    print(f"✅ Engine: Merged raw segments into {len(merged_chunks)} coherent chunks.")

    if not merged_chunks:
        print("No chunks to index.")
        return True

    ids = [f"chunk_{i+1}" for i in range(len(merged_chunks))]
    documents = [chunk["text"] for chunk in merged_chunks]
    metadatas = merged_chunks
    
    print(f"▶️  Engine: Creating embeddings for {len(documents)} merged chunks...")
    embeddings = model.encode(documents, show_progress_bar=True)
    print("✅ Engine: Embeddings created.")

    collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=documents)
    print(f"✅ Engine: Successfully added {collection.count()} documents to the collection.")
    return True

def process_session_pipeline(uploaded_file_path: str, session_folder: str):
    """
    The main processing pipeline that runs all steps and yields progress updates.
    Includes a new step to clean the speaker timeline.
    """
    # Step 1: Prepare Audio
    yield "Step 1/5: Preparing audio from uploaded file..."
    audio_path = prepare_audio_from_upload(uploaded_file_path, session_folder)
    if not audio_path:
        yield "ERROR: Failed to prepare audio file."
        return

    # Step 2: Diarization
    yield "Step 2/5: Identifying speakers..."
    timeline_path = run_diarization(audio_path, session_folder)
    if not timeline_path:
        yield "ERROR: Speaker diarization failed."
        return

    # --- NEW: Step 2.5: Clean the Timeline ---
    yield "Step 2.5/5: Cleaning up speaker timeline..."
    with open(timeline_path, "r") as f:
        raw_timeline = json.load(f)

    cleaned_timeline = _clean_timeline(raw_timeline)

    cleaned_timeline_path = os.path.join(session_folder, "cleaned_timeline.json")
    with open(cleaned_timeline_path, "w") as f:
        json.dump(cleaned_timeline, f, indent=2)

    print(f"✅ Engine: Cleaned timeline. Reduced segments from {len(raw_timeline)} to {len(cleaned_timeline)}.")


    # Step 3: Transcription (now uses the cleaned timeline)
    yield "Step 3/5: Transcribing audio..."
    # Note: We now pass the path to the CLEANED timeline to the transcription function
    transcript_path = run_transcription(audio_path, cleaned_timeline_path, session_folder)
    if not transcript_path:
        yield "ERROR: Transcription failed."
        return

    # Step 4: Indexing
    yield "Step 4/5: Indexing transcript for Q&A..."
    success = run_indexing(transcript_path, session_folder)
    if not success:
        yield "ERROR: Indexing failed."
        return

    yield "✅ Processing complete! You can now ask questions."
# --- NEW: Q&A Function ---
def ask_question(question: str, session_folder: str):
    """
    Answers a question based on the indexed transcript for a session.

    Args:
        question (str): The user's question.
        session_folder (str): The folder for the current session.

    Returns:
        str: The generated answer from the LLM.
    """
    print(f"▶️  Engine: Answering question: '{question}'")

    # 1. Load the embedding model (could be cached in a real app)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

    # 2. Connect to the session-specific ChromaDB
    db_path = os.path.join(session_folder, "chroma_db")
    if not os.path.exists(db_path):
        return "ERROR: The database for this session has not been indexed yet."

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="audio_transcript")

    # 3. Vectorize the question and query the database
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        include=['metadatas', 'documents', 'distances']
    )

    # 4. Fallback if no relevant results are found
    if not results['documents'] or results['distances'][0][0] > 1.3:
        yield "I could not find a relevant answer in the provided source to answer that question."
        return # This stops the function after sending the message

    # 5. Build the context for the prompt
    context = ""
    for i, doc in enumerate(results['documents'][0]):
        metadata = results['metadatas'][0][i]
        speaker = metadata.get('primary_speaker', 'Unknown Speaker')
        context += f"Source {i+1} (Speaker: {speaker} from {metadata['start']:.2f}s to {metadata['end']:.2f}s):\n"
        context += f"\"{doc}\"\n---\n"

    # 6. Construct the prompt
    prompt_template = """
    You are an expert Q&A assistant. Your task is to synthesize a clear and concise answer from the provided context.
    Do not just copy-paste sentences from the sources. Read all the provided information and formulate a new, coherent response in your own words.

    Your response MUST follow these rules:
    1.  Base your answer ONLY on the provided context.
    2.  If the answer is not in the context, state that clearly and DO NOT invent information.
    3.  DO NOT include any citations, source numbers, or phrases like "according to the context" in your answer. Just provide the synthesized answer.

    ---
    CONTEXT:
    {context}
    ---

    USER QUESTION:
    {question}
    """
    final_prompt = prompt_template.format(context=context, question=question)

    # 7. Collect sources and query the LLM

    # Collect unique citation identifiers from the results for programmatic citation
    cited_sources = set()
    for metadata in results.get('metadatas', [[]])[0]:
        cited_sources.add(
            (metadata.get('primary_speaker', 'Unknown'), metadata['start'], metadata['end'])
        )

    # Sort sources by start time for consistent output
    sorted_cited_sources = sorted(list(cited_sources), key=lambda x: x[1])

    # Query Ollama and get the response stream
    response_stream = ollama.chat(
        model="llama3:8b",
        messages=[{'role': 'user', 'content': final_prompt}],
        stream=True
    )

    # First, yield each chunk of the LLM's actual response
    for chunk in response_stream:
        content = chunk['message']['content']
        yield content

    # 8. After the LLM's response is complete, add the programmatic citations
    if sorted_cited_sources:
        yield "\n\n**Sources:**\n" # Add a newline for separation
        for speaker, start, end in sorted_cited_sources:
            yield f"- *{speaker} from {start:.2f}s to {end:.2f}s*\n"
