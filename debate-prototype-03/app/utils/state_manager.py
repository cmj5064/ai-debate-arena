import streamlit as st


def init_session_state():
    # 세션 스테이트 초기화
    if "debate_active" not in st.session_state:
        reset_session_state()


def reset_session_state():
    st.session_state.debate_active = False
    st.session_state.round = 0
