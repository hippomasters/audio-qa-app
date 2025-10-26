import streamlit as st
import os
from datetime import datetime
import engine
import ollama

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

# --- App Title ---
st.title("🎙️ Audio Question-Answering App")
st.markdown("Upload an audio or video file, and I'll create a searchable knowledge base for you to ask questions.")

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

            # **CORRECTION 1: Track success/failure of file processing**
            success_count = 0
            fail_count = 0
            failed_files = []

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

            # **CORRECTION 1 (cont.): Provide accurate feedback to the user**
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
                # **CORRECTION 2: Manually handle streaming to separate answer from sources**
                response_placeholder = st.empty()
                sources_placeholder = st.empty()

                full_response = ""
                sources_text = ""
                is_sources_section = False

                for chunk in engine.ask_question(prompt, session_folder):
                    if chunk == "\n\n**Sources:**\n":
                        is_sources_section = True
                        sources_text += chunk
                        continue

                    if not is_sources_section:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    else:
                        sources_text += chunk
                        sources_placeholder.markdown(sources_text)

                response_placeholder.markdown(full_response) # Final response without cursor

            # **CORRECTION 2 (cont.): Save only the LLM's answer to history**
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.error("It seems the session was lost. Please upload your file again.")
