import torch
import torchaudio
from transformers import pipeline
import json
import time

# --- Configuration ---
AUDIO_FILE = "Test.mp3"
TIMELINE_JSON = "timeline.json"
OUTPUT_JSON = "final_transcript_hf.json"
# We specify the exact model from the Hugging Face Hub
MODEL_ID = "openai/whisper-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def transcribe_with_hf(audio_path: str, timeline_path: str):
    """
    Transcribes audio segments using the Hugging Face Whisper implementation.
    """
    print(f"▶️  Loading HF pipeline for model: '{MODEL_ID}'...")
    # Use the transformers pipeline for ASR
    transcribe_pipeline = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        device=DEVICE
    )
    print("✅ Pipeline loaded.")

    print(f"▶️  Loading and resampling audio file: '{audio_path}'")
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    print("✅ Audio loaded.")

    with open(timeline_path, "r") as f:
        timeline = json.load(f)

    full_transcript = []
    print(f"▶️  Transcribing {len(timeline)} segments...")
    
    for segment in timeline:
        start_time = segment["start"]
        end_time = segment["end"]
        
        start_frame = int(start_time * 16000)
        end_frame = int(end_time * 16000)
        segment_waveform = waveform[:, start_frame:end_frame]
        
        # The pipeline expects a NumPy array
        audio_segment_for_hf = segment_waveform.squeeze().numpy()

        # The pipeline call handles everything. We also pass a parameter to return timestamps.
        result = transcribe_pipeline(audio_segment_for_hf)
        
        full_transcript.append({
            "speaker": segment["speaker"],
            "start": start_time,
            "end": end_time,
            "text": result["text"].strip()
        })

    return full_transcript

if __name__ == "__main__":
    start_time = time.time()
    final_transcript = transcribe_with_hf(AUDIO_FILE, TIMELINE_JSON)
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_transcript, f, indent=2)
        
    print(f"✅ Transcription complete in {time.time() - start_time:.2f}s.")
    print(f"✅ Full transcript saved to '{OUTPUT_JSON}'")
