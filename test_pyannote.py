import torch
from pyannote.audio import Pipeline
import time

# --- Configuration ---
AUDIO_FILE = "jfk.flac"

# --- Main execution ---
if __name__ == "__main__":
    # 1. Check for GPU and select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▶️  Using device: {device}")

    # 2. Load the diarization pipeline
    # This will automatically download the model from Hugging Face Hub using your token.
    print("▶️  Loading speaker diarization pipeline...")
    start_time = time.time()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
    )
    pipeline.to(torch.device(device))
    load_time = time.time() - start_time
    print(f"✅ Pipeline loaded in {load_time:.2f} seconds.")

    # 3. Process the audio file
    print(f"▶️  Analyzing '{AUDIO_FILE}' for speakers...")
    start_time = time.time()
    diarization = pipeline(AUDIO_FILE)
    process_time = time.time() - start_time
    print(f"✅ Diarization finished in {process_time:.2f} seconds.")

    # 4. Print the result
    print("\n" + "="*50)
    print("  SPEAKER DIARIZATION RESULT")
    print("="*50)
    # The 'diarization' object contains all the speaker segments
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"Speaker {speaker} spoke from {turn.start:.2f}s to {turn.end:.2f}s.")
    print("="*50 + "\n")
