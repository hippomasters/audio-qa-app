import ollama
from transformers import pipeline
import torch
import time

# --- Configuration ---
# ‼️ IMPORTANT: Change this to the path of your audio file
AUDIO_FILE_PATH = "Test.mp3" 

# --- Model Configuration ---
WHISPER_MODEL_ID = "openai/whisper-base"
OLLAMA_MODEL = "llama3:8b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Prompt Template for the LLM ---
SUMMARY_PROMPT_TEMPLATE = """
You are an expert assistant specializing in summarizing conversations.
Please provide a concise, well-structured summary of the following transcript.
Focus on the key topics, decisions, and action items discussed.
only give paragraphs for summery max 450 words ,no bullet points
TRANSCRIPT:
"{transcript}"

SUMMARY:
"""

def summarize_audio(audio_path):
    """
    Transcribes an entire audio file and then generates a summary using an LLM.
    """
    start_time = time.time()
    
    # --- Stage 1: Transcription with Whisper ---
    print(f"▶️  Step 1: Transcribing '{audio_path}' with Whisper...")
    
    # Load the ASR pipeline from Hugging Face Transformers
    whisper_pipeline = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_ID,
        device=DEVICE
    )
    
    # Transcribe the entire audio file.
    # chunk_length_s=30 is a recommended setting for longer audio files.
    transcription_result = whisper_pipeline(audio_path, chunk_length_s=30)
    transcript_text = transcription_result["text"]
    
    transcription_time = time.time() - start_time
    print(f"✅ Transcription complete in {transcription_time:.2f}s.")
    print(f"   Transcript preview: \"{transcript_text[:100]}...\"")

    # --- Stage 2: Summarization with Ollama ---
    print("\n▶️  Step 2: Generating summary with Ollama...")
    
    # Format the prompt with the full transcript
    final_prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript_text)
    
    # Send the prompt to the Ollama model and stream the response
    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{'role': 'user', 'content': final_prompt}],
        stream=True
    )
    
    print("\n💡 Summary:")
    # Print the summary as it's being generated
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print("\n")
    
    total_time = time.time() - start_time
    print(f"\n✅ Total process complete in {total_time:.2f}s.")


if __name__ == "__main__":
    summarize_audio(AUDIO_FILE_PATH)
