import streamlit as st

DEFAULT_STATE = {
    "debate_active": False,
    "debate_messages": [],
    "viewing_history": False,
    "loaded_debate_id": None,
    "streaming_active": False,
    "vector_store": None,
    "retrieved_docs": {"pro": [], "con": []},
}


def init_session_state():
    for key, default_value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def set_debate_to_state(topic, messages, docs, debate_id):
    st.session_state.debate_active = True
    st.session_state.debate_messages = messages
    st.session_state.retrieved_docs = docs
    st.session_state.viewing_history = True
    st.session_state.loaded_debate_id = debate_id
    st.session_state.debate_topic = topic


def reset_session_state():
    for key, default_value in DEFAULT_STATE.items():
        st.session_state[key] = default_value
