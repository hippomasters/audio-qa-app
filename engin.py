import engine
import os
import shutil

# --- Test Configuration ---
# Make sure this audio file exists in your project directory
TEST_AUDIO_FILE = "Test.mp3"
# This will be the name of the folder created inside 'sessions/'
TEST_SESSION_ID = "my_first_test_session"

def run_test():
    """
    A simple test harness to run the full engine pipeline.
    """
    print("🚀 STARTING ENGINE TEST 🚀")

    # 1. Set up the session folder
    session_folder = os.path.join("sessions", TEST_SESSION_ID)
    if os.path.exists(session_folder):
        print(f"⚠️  Old test session folder found. Deleting '{session_folder}' to ensure a clean run.")
        shutil.rmtree(session_folder)
    os.makedirs(session_folder)
    print(f"✅ Created new session folder: '{session_folder}'")

    # --- Run the Pipeline Step-by-Step ---

    # Step 1: Diarization
    print("\n--- Running Step 1: Diarization ---")
    timeline_path = engine.run_diarization(TEST_AUDIO_FILE, session_folder)
    if not timeline_path:
        print("❌ TEST FAILED: Diarization step failed.")
        return

    # Step 2: Transcription
    print("\n--- Running Step 2: Transcription ---")
    transcript_path = engine.run_transcription(TEST_AUDIO_FILE, timeline_path, session_folder)
    if not transcript_path:
        print("❌ TEST FAILED: Transcription step failed.")
        return

    # Step 3: Indexing
    print("\n--- Running Step 3: Indexing ---")
    success = engine.run_indexing(transcript_path, session_folder)
    if not success:
        print("❌ TEST FAILED: Indexing step failed.")
        return

    print("\n🎉 ENGINE TEST COMPLETE: All steps ran successfully! 🎉")
    print(f"Check the output files in the '{session_folder}' directory.")

if __name__ == "__main__":
    run_test()
