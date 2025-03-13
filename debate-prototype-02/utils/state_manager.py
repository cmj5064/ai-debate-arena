import streamlit as st


def init_session_state():
    # 세션 스테이트 초기화
    if "debate_active" not in st.session_state:
        reset_session_state()


def reset_session_state():
    st.session_state.debate_active = False
    st.session_state.round = 1
    st.session_state.max_rounds = 1
    st.session_state.debate_history = []
    st.session_state.judge_verdict = None
    st.session_state.current_step = "pro_round_1"
