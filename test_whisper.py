import whisper
import torch
import time

# --- Configuration ---
AUDIO_FILE = "jfk.flac"
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large

# --- Main execution ---
if __name__ == "__main__":
    # 1. Check for GPU and select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▶️  Using device: {device}")

    # 2. Load the Whisper model
    print(f"▶️  Loading Whisper model '{MODEL_SIZE}'...")
    start_time = time.time()
    model = whisper.load_model(MODEL_SIZE, device=device)
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds.")

    # 3. Transcribe the audio file
    print(f"▶️  Transcribing '{AUDIO_FILE}'...")
    start_time = time.time()
    result = model.transcribe(AUDIO_FILE, fp16=torch.cuda.is_available())
    transcribe_time = time.time() - start_time
    print(f"✅ Transcription finished in {transcribe_time:.2f} seconds.")

    # 4. Print the result
    print("\n" + "="*50)
    print("  TRANSCRIPTION RESULT")
    print("="*50)
    print(result["text"].strip())
    print("="*50 + "\n")

