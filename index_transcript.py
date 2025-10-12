import json
import chromadb
from sentence_transformers import SentenceTransformer
import time
from collections import Counter

# --- Configuration ---
TRANSCRIPT_FILE = "final_transcript_hf.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "audio_transcript"

# --- NEW: Advanced Chunk Merging Configuration ---
# The target word count for a chunk. The script will try to create chunks of this size.
TARGET_CHUNK_WORD_COUNT = 500
# The minimum word count for a chunk. Any leftover segments smaller than this will be appended to the last chunk.
MIN_CHUNK_WORD_COUNT = 350

def determine_primary_speaker(chunk_buffer):
    """
    Determines the primary speaker of a chunk based on word count.
    """
    if not chunk_buffer:
        return "Unknown"

    word_counts = Counter()
    for segment in chunk_buffer:
        speaker = segment.get("speaker", "Unknown")
        word_count = len(segment.get("text", "").split())
        word_counts[speaker] += word_count

    # Return the speaker with the most words
    return word_counts.most_common(1)[0][0]

def finalize_chunk(chunk_buffer):
    """
    Finalizes a buffer of segments into a single merged chunk with rich metadata.
    """
    if not chunk_buffer:
        return None

    merged_text = " ".join([seg["text"] for seg in chunk_buffer])
    start_time = chunk_buffer[0]["start"]
    end_time = chunk_buffer[-1]["end"]
    primary_speaker = determine_primary_speaker(chunk_buffer)

    return {
        "primary_speaker": primary_speaker,
        "text": merged_text,
        "start": start_time,
        "end": end_time,
        "original_segments": json.dumps(chunk_buffer) # Store original segments as a JSON string
    }

def index_transcript():
    # 1. Load tools and data
    print(f"▶️  Loading embedding model: '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Model loaded.")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"✅ ChromaDB collection '{COLLECTION_NAME}' ready.")

    with open(TRANSCRIPT_FILE, "r") as f:
        transcript_data = json.load(f)
    print(f"▶️  Loaded {len(transcript_data)} raw segments from '{TRANSCRIPT_FILE}'.")

    # 2. --- NEW: Advanced Chunk Merging Logic ---
    merged_chunks = []
    chunk_buffer = []

    for i, segment in enumerate(transcript_data):
        chunk_buffer.append(segment)
        current_words = len(" ".join([s["text"] for s in chunk_buffer]).split())

        # Check if the chunk is "full" based on word count
        # or if it's the last segment in the transcript
        if current_words >= TARGET_CHUNK_WORD_COUNT or i == len(transcript_data) - 1:
            merged_chunks.append(finalize_chunk(chunk_buffer))
            chunk_buffer = [] # Reset the buffer

    # Handle any leftover small chunks at the end
    if chunk_buffer:
         # If the last chunk is too small, merge it with the previous one
        if merged_chunks and len(" ".join([s['text'] for s in chunk_buffer]).split()) < MIN_CHUNK_WORD_COUNT:
            last_main_chunk = merged_chunks[-1]
            last_buffer_as_chunk = finalize_chunk(chunk_buffer)

            # Combine text and update end time
            last_main_chunk['text'] += " " + last_buffer_as_chunk['text']
            last_main_chunk['end'] = last_buffer_as_chunk['end']

            # Combine original segments metadata
            original_segments = json.loads(last_main_chunk['original_segments'])
            original_segments.extend(chunk_buffer)
            last_main_chunk['original_segments'] = json.dumps(original_segments)

        else: # Otherwise, add it as its own chunk
            merged_chunks.append(finalize_chunk(chunk_buffer))


    print(f"✅ Merged raw segments into {len(merged_chunks)} coherent chunks.")

    # 3. Prepare and add to ChromaDB
    ids = [f"chunk_{i+1}" for i in range(len(merged_chunks))]
    documents = [chunk["text"] for chunk in merged_chunks]
    metadatas = merged_chunks

    if documents:
        print(f"▶️  Creating embeddings for {len(documents)} merged chunks...")
        embeddings = model.encode(documents, show_progress_bar=True)
        print("✅ Embeddings created.")

        print("▶️  Adding documents to the ChromaDB collection...")
        collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=documents)
        print(f"✅ Successfully added {collection.count()} documents to the collection.")
    else:
        print("No documents to index.")

if __name__ == "__main__":
    start_time = time.time()
    index_transcript()
    print(f"✅ Indexing complete in {time.time() - start_time:.2f}s.")
