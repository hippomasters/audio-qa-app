import chromadb
import ollama
from sentence_transformers import SentenceTransformer
import time
import json # Added for safely parsing metadata

# --- Configuration ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "audio_transcript"
OLLAMA_MODEL = "llama3:8b"
SIMILARITY_THRESHOLD = 5.0 # Lower is stricter. 1.0 is a good starting point.

# --- Updated Prompt Template ---
PROMPT_TEMPLATE = """
You are a factual assistant. Your sole task is to accurately answer the user's question based ONLY on the provided context.

Instructions:
- Your answer must be synthesized from the information in the context sources.
- Do add any conversational text or introductions like "Based on the context...".
- Respond using only information found in the context. If the answer is not in the context, you MUST say "I could not find an answer in the provided transcript."
- After your answer, you MUST cite the sources you used. Use the format.
- do not answer if not found in content
Example of a perfect response:
The budget was approved for the Q4 campaign.

---
CONTEXT:
{context}
---

USER QUESTION:
{question}
"""

def main():
    """
    The main function for the interactive QA application.
    """
    # 1. Load tools
    print("▶️  Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Embedding model loaded.")

    print("▶️  Connecting to ChromaDB...")
    collection = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_or_create_collection(name=COLLECTION_NAME)
    print("✅ ChromaDB connection successful.")

    print("\n✅ All tools loaded. You can now ask questions about your audio.")
    print("   Type 'quit' or 'exit' to stop.")

    # 2. Start interactive loop
    while True:
        question = input("\n> Ask a question: ")
        if question.lower() in ['quit', 'exit']:
            print("Exiting application. Goodbye!")
            break

        # 3. Vectorize the question
        question_embedding = embedding_model.encode(question).tolist()

        # 4. Search the database
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=2,
            include=['metadatas', 'documents', 'distances']
        )
        # ⬇️ ADD THIS DEBUGGING LINE ⬇️
        if results['distances']:
            print(f"DEBUG: Best match distance score is {results['distances'][0][0]}")


        # 5. Fallback if results are not relevant
        if not results['documents'] or results['distances'][0][0] > SIMILARITY_THRESHOLD:
            print("\n💡 Answer:\nI could not find a relevant answer in the provided transcript.")
            continue

        # 6. Build the context string
        context = ""
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                # --- FIX: Use the new 'primary_speaker' key ---
                speaker = metadata.get('primary_speaker', 'Unknown Speaker')
                context += f"Source {i+1} (Speaker: {speaker} from {metadata['start']:.2f}s to {metadata['end']:.2f}s):\n"
                context += f"\"{doc}\"\n---\n"

        # 7. Construct the final prompt
        final_prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # 8. Query the LLM
        print("\n💡 Answer:")
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': final_prompt}],
            stream=True
        )
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print("\n")


if __name__ == "__main__":
    main()
