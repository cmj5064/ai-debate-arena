import streamlit as st
from typing import Dict, List

from utils.state_manager import reset_session_state
from workflow.graph import create_debate_graph
from workflow.state import DebateState, AgentType
from langfuse.callback import CallbackHandler


# 토론 시작 뷰
def render_start_view():
    if st.button("토론 시작"):
        start_debate()


# 토론 시작 처리
def start_debate():

    # 세션 상태 초기화
    st.session_state.vector_store = None
    st.session_state.debate_messages = []
    st.session_state.streaming_active = True

    # 사용자 입력값 가져오기
    debate_topic = st.session_state.ui_debate_topic
    max_rounds = st.session_state.ui_max_rounds

    # with st.container():
    st.header(f"토론 주제: {debate_topic}")
    st.subheader("토론 진행 상황")

    pro_placeholder = st.empty()
    con_placeholder = st.empty()
    judge_placeholder = st.empty()

    debate_graph = create_debate_graph()

    initial_state: DebateState = {
        "topic": debate_topic,
        "messages": [],
        "current_round": 1,
        "max_rounds": max_rounds,
        "current_speaker": AgentType.PRO,
        "ui_placeholders": {
            "pro_agent_container": pro_placeholder,
            "con_agent_container": con_placeholder,
            "judge_container": judge_placeholder,
        },
    }

    with st.spinner("토론이 진행 중입니다..."):

        final_result = initial_state.copy()
        langfuse_handler = CallbackHandler()
        for chunk in debate_graph.stream(
            initial_state, config={"callbacks": [langfuse_handler]}
        ):
            if chunk:
                for key, value in chunk.items():
                    final_result[key] = value

        if "messages" not in final_result:
            final_result["messages"] = []

        st.session_state.debate_messages = final_result["messages"]
        st.session_state.debate_active = True
        st.session_state.streaming_active = False
        st.session_state.viewing_history = False

    st.rerun()


def render_debate_view():

    debate_topic = st.session_state.ui_debate_topic
    st.header(f"토론 주제: {debate_topic}")
    st.header("토론 결과")

    messages = st.session_state.debate_messages
    render_debate_messages(messages)

    render_judge_evaluation(messages)

    render_control_buttons()


def render_debate_messages(messages: List[Dict[str, str]]):
    """토론 메시지 렌더링"""
    total_rounds = len([m for m in messages if m["role"] == AgentType.PRO])

    # 라운드별로 그룹화하여 표시
    for round_num in range(1, total_rounds + 1):
        st.subheader(f"라운드 {round_num}")

        # 이 라운드의 찬성측 메시지
        pro_index = (round_num - 1) * 2
        if pro_index < len(messages) and messages[pro_index]["role"] == AgentType.PRO:
            with st.container(border=True):
                st.markdown("**⭕ 찬성 측:**")
                st.write(messages[pro_index]["content"])

        # 이 라운드의 반대측 메시지
        con_index = (round_num - 1) * 2 + 1
        if con_index < len(messages) and messages[con_index]["role"] == AgentType.CON:
            with st.container(border=True):
                st.markdown("**❌ 반대 측:**")
                st.write(messages[con_index]["content"])

        st.divider()


def render_judge_evaluation(messages: List[Dict[str, str]]):
    """심판 평가 렌더링"""
    if messages and messages[-1]["role"] == AgentType.JUDGE:
        with st.container(border=True):
            st.subheader("🧑‍⚖️ 최종 평가")
            st.write(messages[-1]["content"])


def render_control_buttons():
    """제어 버튼 렌더링"""

    if st.button("새 토론 시작", use_container_width=True):
        reset_session_state()
        st.rerun()
