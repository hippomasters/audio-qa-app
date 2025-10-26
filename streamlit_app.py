import streamlit as st
import os
from datetime import datetime
import engine
import ollama
import json
import shutil

def get_session_list():
    """Lists the names of existing session folders."""
    if not os.path.exists("sessions"):
        os.makedirs("sessions") # Ensure the base directory exists
        return []

    # List only directories, ignore files
    try:
        session_folders = [
            f for f in os.listdir("sessions")
            if os.path.isdir(os.path.join("sessions", f))
        ]
        # Sort by name (which is date-time) for chronological order
        session_folders.sort(reverse=True)
        return session_folders
    except OSError as e:
        st.error(f"Error reading session directory: {e}")
        return []

def save_chat_history(session_folder: str, messages: list):
    """Saves the chat history to a JSON file in the session folder."""
    history_file = os.path.join(session_folder, "chat_history.json")
    try:
        with open(history_file, "w") as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

def load_chat_history(session_folder: str):
    """Loads the chat history from a JSON file in the session folder."""
    history_file = os.path.join(session_folder, "chat_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
            return []
    else:
        return [] # Return empty list if no history file exists

# --- Page Configuration ---
st.set_page_config(
    page_title="Audio QA App",
    page_icon="🎙️",
    layout="wide"
)


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "session_folder" not in st.session_state:
    # --- MODIFIED: Load the most recent session on startup ---
    session_list = get_session_list() # Uses the helper function we added
    if session_list: # If there are past sessions
        most_recent_session = os.path.join("sessions", session_list[0]) # Assumes list is sorted newest first
        st.session_state.session_folder = most_recent_session
        st.session_state.messages = load_chat_history(most_recent_session)
        # Determine processed files for the loaded session
        processed_files_in_session = []
        try:
            for item in os.listdir(most_recent_session):
                 if item.endswith(('.mp3', '.wav', '.m4a', '.mp4', '.mov', '.avi', '.mkv')):
                     processed_files_in_session.append(item)
        except OSError as e:
             print(f"Error reading initial session folder {most_recent_session}: {e}") # Use print for startup
        st.session_state.processed_files = processed_files_in_session
        st.session_state.processing_complete = bool(processed_files_in_session)
    else: # If no past sessions, initialize as None
        st.session_state.session_folder = None
        st.session_state.messages = []
        st.session_state.processed_files = []
        st.session_state.processing_complete = False
    # --- End of Modification ---


# --- Sidebar for Session Management ---
with st.sidebar:
    st.header("Conversations")

    # Button to start a completely new session
    if st.button("➕ New Chat"):
        # Reset all relevant session state variables
        st.session_state.session_folder = None
        st.session_state.processed_files = []
        st.session_state.messages = []
        st.session_state.processing_complete = False
        st.rerun() # Rerun the app to reflect the reset state

    st.markdown("---") # Separator

    # List existing sessions
    session_list = get_session_list()
    if not session_list:
        st.caption("No past conversations found.")
    else:
        st.caption("Select a past conversation:")
        for session_name in session_list:
            # --- MODIFIED: Use columns for Load and Delete buttons ---
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                # Load Button (same logic as before)
                if st.button(session_name, key=f"session_{session_name}", use_container_width=True):
                    st.session_state.session_folder = os.path.join("sessions", session_name)
                    st.session_state.messages = load_chat_history(st.session_state.session_folder)
                    processed_files_in_session = []
                    try:
                        for item in os.listdir(st.session_state.session_folder):
                            if item.endswith(('.mp3', '.wav', '.m4a', '.mp4', '.mov', '.avi', '.mkv')):
                                processed_files_in_session.append(item)
                    except OSError as e:
                        st.error(f"Error reading session folder {session_name}: {e}")
                    st.session_state.processed_files = processed_files_in_session
                    st.session_state.processing_complete = bool(processed_files_in_session)
                    st.rerun()
            with col2:
                # Delete Button
                if st.button("🗑️", key=f"delete_{session_name}", help="Delete this session"):
                    try:
                        folder_to_delete = os.path.join("sessions", session_name)
                        shutil.rmtree(folder_to_delete)
                        # If the deleted session was the active one, reset the state
                        if st.session_state.session_folder == folder_to_delete:
                            st.session_state.session_folder = None
                            st.session_state.processed_files = []
                            st.session_state.messages = []
                            st.session_state.processing_complete = False
                        st.success(f"Deleted session: {session_name}")
                        st.rerun() # Refresh the sidebar list
                    except Exception as e:
                        st.error(f"Error deleting session {session_name}: {e}")
            # --- End of Modification ---

    # --- ADDED: Rename functionality for the CURRENT session ---
    if st.session_state.session_folder:
        st.markdown("---")
        st.subheader("Rename Current Session")
        current_name = os.path.basename(st.session_state.session_folder)
        new_name = st.text_input("New name:", value=current_name, key="rename_input")
        if st.button("Rename", key="rename_button"):
            if new_name and new_name != current_name:
                try:
                    new_folder_path = os.path.join("sessions", new_name)
                    if not os.path.exists(new_folder_path):
                        os.rename(st.session_state.session_folder, new_folder_path)
                        st.session_state.session_folder = new_folder_path # Update session state
                        st.success(f"Renamed session to: {new_name}")
                        st.rerun() # Refresh sidebar and display
                    else:
                        st.error("A session with that name already exists.")
                except Exception as e:
                    st.error(f"Error renaming session: {e}")
            elif not new_name:
                st.error("New name cannot be empty.")
    # -----------------------------------------------------------

# --- App Title (Now outside the sidebar) ---
st.title("🎙️ Audio Question-Answering App")

# Display current session info (optional but helpful)
if st.session_state.session_folder:
    st.caption(f"Current session: `{os.path.basename(st.session_state.session_folder)}`")
else:
     st.caption("Start a new chat by uploading or recording audio.")


# --- Input Section ---
st.header("1. Provide Your Audio or Video File")

uploaded_files = st.file_uploader(
    "Choose files (.mp3, .wav, .mp4, etc.)",
    type=['mp3', 'wav', 'm4a', 'mp4', 'mov', 'avi', 'mkv'],
    accept_multiple_files=True
)

# --- File Processing Logic ---
if uploaded_files:
    new_files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if new_files_to_process:
        if st.button(f"Process {len(new_files_to_process)} New File(s)", key="process_upload_button"):
            if "session_folder" not in st.session_state or st.session_state.session_folder is None:
                session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                st.session_state.session_folder = os.path.join("sessions", session_id)
                os.makedirs(st.session_state.session_folder, exist_ok=True)

            file_paths_to_process = []
            for new_file in new_files_to_process:
                path = os.path.join(st.session_state.session_folder, new_file.name)
                with open(path, "wb") as f:
                    f.write(new_file.getbuffer())
                file_paths_to_process.append(path)
                st.session_state.processed_files.append(new_file.name)

            # Track success/failure
            success_count = 0
            fail_count = 0
            failed_files = []

            # --- ADDED: Clear GPU cache before processing ---
            engine.clear_gpu_cache()
            # --------------------------------------------------

            progress_bar = st.empty()
            with st.spinner("Processing files... This may take a few minutes."):
                for status in engine.process_session_pipeline(file_paths_to_process, st.session_state.session_folder):
                    progress_bar.text(status)
                    if "ERROR:" in status:
                        fail_count += 1
                        try:
                            failed_file = status.split("'")[1]
                            failed_files.append(failed_file)
                        except IndexError:
                            pass

            # Provide accurate feedback
            total_processed = len(file_paths_to_process)
            success_count = total_processed - fail_count
            if fail_count == 0 and success_count > 0:
                st.success(f"Successfully processed {success_count} file(s)!")
                st.session_state.processing_complete = True
            elif success_count > 0 and fail_count > 0:
                st.warning(f"Processed {success_count} file(s), but failed to process {fail_count} file(s): {', '.join(failed_files)}")
                st.session_state.processing_complete = True
            elif fail_count > 0 and success_count == 0:
                st.error(f"Failed to process all {fail_count} file(s): {', '.join(failed_files)}")
                st.session_state.processing_complete = False

# --- Chat Interface (Only shows after processing is complete) ---
if st.session_state.get('processing_complete', False):
    st.header("2. Chat with Your Audio")
    with st.expander("Show Full Transcript"):
        transcript_file_path = os.path.join(st.session_state.session_folder, "final_transcript_hf.json")
        if os.path.exists(transcript_file_path):
            try:
                with open(transcript_file_path, "r") as f:
                    transcript_data = json.load(f)

                # Display transcript data (simple version)
                transcript_text = ""
                for segment in transcript_data:
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    speaker = segment.get('primary_speaker', segment.get('speaker', 'Unknown')) # Handle both old and new keys
                    text = segment.get('text', '')
                    source = segment.get('source_file', '')
                    transcript_text += f"**[{speaker} {start_time:.2f}s - {end_time:.2f}s | {source}]**\n{text}\n\n"

                st.markdown(transcript_text)
            except Exception as e:
                st.error(f"Error loading or displaying transcript: {e}")
        else:
            st.warning("Transcript file not found for this session.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your audio..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if 'session_folder' in st.session_state:
            session_folder = st.session_state['session_folder']
            with st.chat_message("assistant"):
                response_placeholder = st.empty() # Use one placeholder for combined output

                full_response_content = "" # Accumulate everything yielded
                llm_answer_part = ""     # Accumulate only the LLM's answer part
                sources_part = ""        # Accumulate only the sources part
                is_sources_section = False

                for chunk in engine.ask_question(prompt, session_folder):
                    full_response_content += chunk # Add chunk to the full content
                    if chunk == "\n\n**Sources:**\n":
                        is_sources_section = True

                    if not is_sources_section:
                        llm_answer_part += chunk # Accumulate only the answer

                    # Update the display with accumulated content + typing cursor
                    response_placeholder.markdown(full_response_content + "▌")

                response_placeholder.markdown(full_response_content) # Final display without cursor

            # **CORRECTION:** Save the complete markdown content (answer + sources) to history
            st.session_state.messages.append({"role": "assistant", "content": full_response_content})

            # Save the updated history (as added in Action 1.3)
            save_chat_history(session_folder, st.session_state.messages)
        else:
            st.error("It seems the session was lost. Please upload your file again.")
