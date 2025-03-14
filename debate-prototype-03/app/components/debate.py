import streamlit as st

from components.options import render_options
from workflow.graph import create_debate_graph
from workflow.state import DebateState, AgentType
from langfuse.callback import CallbackHandler


# 토론 시작 뷰
def render_start_view():

    st.markdown(
        """
        ### 프로젝트 소개
        이 애플리케이션은 LangGraph를 활용하여 AI 에이전트 간의 토론 워크플로우를 구현합니다.
        찬성 측, 반대 측, 그리고 심판 역할의 AI가 주어진 주제에 대해 체계적으로 토론을 진행합니다.
        """
    )
    render_options()

    if st.button("토론 시작"):
        start_debate()


# 토론 시작 처리
def start_debate():

    # 사용자 입력값 가져오기
    debate_topic = st.session_state.ui_debate_topic
    max_rounds = st.session_state.ui_max_rounds

    # 그래프 생성
    debate_graph = create_debate_graph()

    # 초기 상태 설정
    initial_state: DebateState = {
        "topic": debate_topic,
        "messages": [],
        "current_round": 1,
        "max_rounds": max_rounds,
        "current_speaker": AgentType.PRO,
    }

    # 토론 시작
    with st.spinner("토론이 진행 중입니다... 완료까지 잠시 기다려주세요."):

        langfuse_handler = CallbackHandler()
        result = debate_graph.invoke(
            initial_state, config={"callbacks": [langfuse_handler]}
        )

        # 결과를 세션 스테이트에 저장
        st.session_state.debate_messages = result["messages"]
        st.session_state.debate_active = True

    # 페이지 새로고침하여 결과 표시
    st.rerun()


def render_debate_view():
    # 토론 주제 표시
    debate_topic = st.session_state.ui_debate_topic
    st.header(f"토론 주제: {debate_topic}")

    messages = st.session_state.debate_messages
    total_rounds = len([m for m in messages if m["role"] == AgentType.PRO])

    # 라운드별로 그룹화하여 표시
    for round_num in range(1, total_rounds + 1):
        st.subheader(f"라운드 {round_num}")

        # 이 라운드의 찬성측 메시지 찾기 (인덱스는 (라운드-1)*2)
        pro_index = (round_num - 1) * 2
        if pro_index < len(messages) and messages[pro_index]["role"] == AgentType.PRO:
            st.markdown("**🙆🏻‍♀️ 찬성 측**")
            st.write(messages[pro_index]["content"])

        # 이 라운드의 반대측 메시지 찾기 (인덱스는 (라운드-1)*2 + 1)
        con_index = (round_num - 1) * 2 + 1
        if con_index < len(messages) and messages[con_index]["role"] == AgentType.CON:
            st.markdown("**🙅🏻‍♂️ 반대 측**")
            st.write(messages[con_index]["content"])

        st.divider()

    # 심판 평가 표시 (마지막 메시지)
    if messages and messages[-1]["role"] == AgentType.JUDGE:
        st.subheader("👩🏼‍⚖️ 최종 평가")
        st.write(messages[-1]["content"])

    def reset_debate():
        st.session_state.debate_active = False
        st.session_state.debate_messages = []

    st.button("새 토론 시작", on_click=reset_debate, use_container_width=True)
