import streamlit as st
import os
from datetime import datetime
import engine
import ollama


# --- Page Configuration ---
# This must be the first Streamlit command.
st.set_page_config(
    page_title="Audio QA App",
    page_icon="🎙️",
    layout="wide"
)

# --- Session State Initialization ---
# This comes AFTER the page config.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- App Title ---
st.title("🎙️ Audio Question-Answering App")
st.markdown("Upload an audio or video file, and I'll create a searchable knowledge base for you to ask questions.")

# --- File Uploader ---
st.header("1. Upload Your Audio or Video File")

uploaded_file = st.file_uploader(
    "Choose a file (.mp3, .wav, .mp4, .mov...)",
    type=['mp3', 'wav', 'm4a', 'mp4', 'mov', 'avi', 'mkv']
)

# --- Process Button and Logic ---
st.header("2. Process and Ask Questions")

# This logic handles saving the file immediately upon upload
# and storing its path in the session state (the app's memory).
if uploaded_file is not None:
    # Check if this is a new file upload
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        # Create a unique session folder as soon as a new file is uploaded
        session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_folder = os.path.join("sessions", session_id)
        os.makedirs(session_folder, exist_ok=True)

        # Save the uploaded file immediately
        file_path = os.path.join(session_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Store the new file info in the app's memory
        st.session_state['processed_file_path'] = file_path
        st.session_state['session_folder'] = session_folder
        st.session_state['current_file'] = uploaded_file.name

        st.info(f"File ready for processing: `{uploaded_file.name}`")

# This "Process" button logic is now separate and relies on the session state.
if st.button("Process Audio/Video"):
    # Check the session state to make sure a file has been uploaded and saved
    if 'processed_file_path' in st.session_state:
        file_to_process = st.session_state['processed_file_path']
        folder_to_process = st.session_state['session_folder']

        st.info(f"Starting processing for: {os.path.basename(file_to_process)}")

        # Call the engine and display real-time progress
        progress_bar = st.empty()
        with st.spinner("Processing in progress... This may take a few minutes."):
            for status in engine.process_session_pipeline(file_to_process, folder_to_process):
                progress_bar.text(status)

        st.success("Processing complete!")
    else:
        st.error("Please upload a file first before processing.")
# --- Chat Interface ---
st.header("3. Chat with Your Audio")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your audio..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the current session folder from the app's memory
    if 'session_folder' in st.session_state:
        session_folder = st.session_state['session_folder']

        # Display assistant response
        with st.chat_message("assistant"):
            # Use st.write_stream to display the streamed response from the engine
            response_stream = engine.ask_question(prompt, session_folder)
            full_response = st.write_stream(response_stream)

        # Add the complete assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.error("It seems the session was lost. Please upload your file again.")
