import torch
from pyannote.audio import Pipeline
import time
import json

# --- Configuration ---
AUDIO_FILE = "Test.mp3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_JSON = "timeline.json"

def create_speaker_timeline(audio_path: str):
    """
    Runs speaker diarization and saves the timeline to a JSON file.
    """
    print("▶️  Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device(DEVICE))
    print("✅ Pipeline loaded.")

    print(f"▶️  Analyzing '{audio_path}' for speakers...")
    diarization = pipeline(audio_path)
    print("✅ Diarization complete.")

    timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        timeline.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        })

    return timeline

if __name__ == "__main__":
    speaker_timeline = create_speaker_timeline(AUDIO_FILE)
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(speaker_timeline, f, indent=2)
        
    print(f"✅ Speaker timeline has been saved to '{OUTPUT_JSON}'")
