import chromadb
import pandas as pd

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "audio_transcript"

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

# Fetch everything
results = collection.get(include=["documents", "metadatas", "embeddings"])

# Convert to DataFrame
df = pd.DataFrame({
    "id": results["ids"],
    "document": results["documents"],
    "metadata": results["metadatas"],
    "embedding_len": [len(e) for e in results["embeddings"]],
})

# Save to CSV
df.to_csv("chroma_dump.csv", index=False)
print("✅ Dumped to chroma_dump.csv")

