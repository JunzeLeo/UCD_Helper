import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from google_ASR import transcribe_file_with_auto_punctuation
from audio.utils import chain_with_message_history

st.title("History Helper")
DATA_DIR = "exp_data"

# set containers for layout
function_block = st.container()
col1, col2 = st.columns(2)
with col1:
    chat_history_block = st.container()
with col2:
    analysis_block = st.container()

# initiliaze session state and "need analysis"

if "need_analysis" not in st.session_state:
    st.session_state["need_analysis"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "analysis" not in st.session_state:
    st.session_state["analysis"] = {x: [] for x in ["HEAR, SEE, SAY, DO, THINK & FEEL"]}


def display_all_messages():
    with chat_history_block:
        for message in st.session_state.messages:
            st.chat_message(name=message["role"]).write(message["content"])


with st.sidebar:
    user_id = st.text_input(
        label="Please input your User ID here:",
        key="user_id",
        type="default",
        value=None,
    )
    session_id = st.text_input(
        label="Please input your Session ID here ↓ Format: (\w+)_(\d)",
        key="session_id",
        type="default",
        value=None,
    )
    history_button = st.button(
        label="Show Conversation History",
        key="history_button",
        on_click=display_all_messages,
        disabled=not st.session_state.messages,
    )


if __name__ == "__main__":
    if not user_id:
        st.warning("Please input your User ID on the left pane to start conversation.")
        st.stop()
    if not session_id:
        st.warning(
            "Please input your Session ID on the left pane to start conversation. Format: (\w+)_(\d)"
        )
        st.stop()
    st.session_state["session_id"] = session_id
    if not os.path.exists(os.path.join(DATA_DIR, user_id)):
        os.makedirs(os.path.join(DATA_DIR, user_id))

    with function_block:
        st.info("Please allow the browser to access your microphone.")
        st.info(
            "Please click the microphone button to start conversation. Start talking when the microphone turns red. You may start the conversation with greetings like '你好' or 'Hello' to the chatbot."
        )
        st.info("Please speak in English.")
        audio_bytes = audio_recorder(
            sample_rate=44100, pause_threshold=1, auto_start=False
        )

    if audio_bytes:
        if not hasattr(st.session_state, "file_index"):
            st.session_state.file_index = 1
        file_index = st.session_state.file_index
        # save to file and transcribe (auto increment file name by 1)
        with open(os.path.join(DATA_DIR, user_id, f"{file_index}.wav"), "wb") as f:
            f.write(audio_bytes)

        response = transcribe_file_with_auto_punctuation(audio_bytes)
        first_alternative = (
            response.results[0].alternatives[0].transcript
            if response.results
            else "N.A."
        )

        with chat_history_block:
            st.chat_message(name="user").write(first_alternative)

        # update to session state
        st.session_state.messages.append(
            {
                "role": "user",
                "content": first_alternative,
            }
        )
        st.session_state.file_index += 1
        st.session_state["need_analysis"] = True

        # Display analysis result
        with analysis_block:
            with st.spinner("System Processing..."):
                # Do analysis
                analysis_result = chain_with_message_history.invoke(
                    {"input": first_alternative},
                    {"configurable": {"session_id": session_id}},
                )
            st.write(analysis_result)

        # update to analysis in session state
        for k, v in analysis_result.items():
            st.session_state.analysis[k].append(v)
